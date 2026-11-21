/// \file modeling_lfm3.cpp
/// \brief LFM3 class
/// \note LiquidAI LFM3: ChatML, `<think>` opened by the template when thinking is on, Qwen-style XML tool calls.

#include "AutoModel/modeling_lfm3.hpp"
#include "models/lfm3/flm/aie2p/lfm3_npu.hpp"
#include "utils/utils.hpp"
#include "utils/npu_platform.hpp"
#include "metrices.hpp"
#include "base64.hpp"

/************              LFM3 family            **************/
LFM3::LFM3(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "LFM3") {}

void LFM3::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    if (utils::get_device() == utils::npu_platform::aie2) {
        // If running on XDNA 1 / AIE2 (Hawk Point/Phoenix) and native NPU1 overlays are not present,
        // safely fallback to CPU AVX2 acceleration to prevent 8-column NPU2 ERT hardware timeouts.
        const std::string exe_dir = utils::get_executable_directory();
        const std::string m_name = model_info.value("name", "LFM3-1.7B-NPU2");
        const std::string xclbin_dir = utils::path_join(exe_dir, "xclbins", m_name);
        const bool has_npu1 = std::filesystem::exists(utils::path_join(xclbin_dir, "layer_npu1.xclbin"));
        if (!has_npu1) {
#if defined(_WIN32)
            if (!std::getenv("LFM3_PREFILL")) _putenv_s("LFM3_PREFILL", "host");
            if (!std::getenv("LFM3_DECODE")) _putenv_s("LFM3_DECODE", "host");
#else
            if (!std::getenv("LFM3_PREFILL")) setenv("LFM3_PREFILL", "host", 0);
            if (!std::getenv("LFM3_DECODE")) setenv("LFM3_DECODE", "host", 0);
#endif
        }
    }
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();
    this->enable_think = true;   // the upstream template thinks by default

    // LiquidAI's recommendation for chat, tool use and agents: temperature 0.1, top_k 50, repetition_penalty 1.1
    // (no top_p / min_p filtering)
    sampler_config config;
    config.top_k = 50;
    config.top_p = 1.0f;
    config.min_p = 0.0f;
    config.temperature = 0.1f;
    config.rep_penalty = 1.1f;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
    _prefix_attach(lm_engine, lfm3_npu_state_api(lm_engine));
}

void LFM3::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
    this->tokenizer->is_doubled_encoded = true;   // byte-level BPE, like LFM2
}

std::string LFM3::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools, bool continue_final_message) {
    minja::chat_template_inputs inputs;
    minja::chat_template_options opt;
    opt.polyfill_tool_responses = false;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->tools_enabled())
        inputs.tools = tools;
    return this->_render_chat_template(inputs, continue_final_message, opt);
}

bool LFM3::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already formatted messages, usually from the REST API
        templated_text = this->apply_chat_template(input.messages, input.tools, meta_info.continue_final_message);
    }
    else { // a plain text prompt, usually from the CLI
        nlohmann::ordered_json messages;
        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // Prefix slots. The generation prompt ("<|im_start|>assistant\n<think>" or "...<think></think>") is how the next turn
    // renders this reply's header (a token prefix of it either way): save the whole prompt. Boundary marker "<|im_start|>user".
    if (this->_prefix_slot_mode(meta_info, tokens.size(), 0)) {
        return this->_prefix_text_insert(meta_info, tokens, 0, "<|im_start|>user", is_cancelled);
    }
    this->_prefix_legacy_path();

    return this->_shared_insert(meta_info, tokens, is_cancelled);
}

std::string LFM3::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string LFM3::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    return this->generate(meta_info, length_limit, os);
}

/************              LFM3-VL family            **************/
LFM3_VL::LFM3_VL(flm_rt::device* npu_device_inst) : LFM3(npu_device_inst) {}

bool LFM3_VL::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (!input.images.empty()) {
        for (const auto& img_source : input.images) {
            image_data_t img;
            bool ok = false;
            if (img_source.rfind("data:", 0) == 0 || img_source.size() > 512) {
                ok = image_reader_.load_image_base64(img_source, img);
            } else {
                ok = image_reader_.load_image(img_source, img);
            }
            if (ok) {
                if (input.prompt.find("<|image|>") == std::string::npos) {
                    input.prompt = "<|image|><|begin_of_image|>" + input.prompt;
                }
            }
        }
    }
    return LFM3::insert(meta_info, input, is_cancelled);
}

/************              LFM3-Audio family            **************/
LFM3_Audio::LFM3_Audio(flm_rt::device* npu_device_inst) : LFM3(npu_device_inst) {}

bool LFM3_Audio::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (!input.audios.empty()) {
        for (const auto& aud_source : input.audios) {
            audio_data_t aud;
            bool ok = false;
            if (aud_source.rfind("data:", 0) == 0 || aud_source.size() > 512) {
                std::string raw = base64::from_base64(aud_source);
                ok = audio_reader_.load_audio_from_memory(reinterpret_cast<const uint8_t*>(raw.data()), raw.size(), aud);
            } else {
                ok = audio_reader_.load_audio(aud_source, aud);
            }
            if (ok) {
                if (input.prompt.find("<|audio|>") == std::string::npos) {
                    input.prompt = "<|audio|><|begin_of_audio|>" + input.prompt;
                }
            }
        }
    }
    return LFM3::insert(meta_info, input, is_cancelled);
}

/************              LFM3-Omni family            **************/
LFM3_Omni::LFM3_Omni(flm_rt::device* npu_device_inst) : LFM3(npu_device_inst) {}

bool LFM3_Omni::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    if (!input.images.empty()) {
        for (const auto& img_source : input.images) {
            image_data_t img;
            bool ok = false;
            if (img_source.rfind("data:", 0) == 0 || img_source.size() > 512) {
                ok = image_reader_.load_image_base64(img_source, img);
            } else {
                ok = image_reader_.load_image(img_source, img);
            }
            if (ok && input.prompt.find("<|image|>") == std::string::npos) {
                input.prompt = "<|image|>" + input.prompt;
            }
        }
    }
    if (!input.audios.empty()) {
        for (const auto& aud_source : input.audios) {
            audio_data_t aud;
            bool ok = false;
            if (aud_source.rfind("data:", 0) == 0 || aud_source.size() > 512) {
                std::string raw = base64::from_base64(aud_source);
                ok = audio_reader_.load_audio_from_memory(reinterpret_cast<const uint8_t*>(raw.data()), raw.size(), aud);
            } else {
                ok = audio_reader_.load_audio(aud_source, aud);
            }
            if (ok && input.prompt.find("<|audio|>") == std::string::npos) {
                input.prompt = "<|audio|>" + input.prompt;
            }
        }
    }
    return LFM3::insert(meta_info, input, is_cancelled);
}
