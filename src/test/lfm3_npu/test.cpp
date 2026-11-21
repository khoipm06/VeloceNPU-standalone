#include <iostream>
#include <cmath>
#define NOMINMAX
#include "utils/utils.hpp"
#include "utils/vm_args.hpp"
#include "AutoModel/modeling_lfm3.hpp"
#include "model_list.hpp"
#include "utils/vm_args.hpp"
#include "metrices.hpp"

flm_rt::device npu_device_global;

// Model-specific factory function for LFM3 family
inline std::pair<std::string, std::unique_ptr<AutoModel>> get_lfm3_model(const std::string& model_tag) {
    if (model_tag == "lfm3-vl" || model_tag == "lfm3-vl:1.7b") {
        return {"lfm3-vl:1.7b", std::make_unique<LFM3_VL>(&npu_device_global)};
    }
    if (model_tag == "lfm3-audio" || model_tag == "lfm3-audio:1.7b") {
        return {"lfm3-audio:1.7b", std::make_unique<LFM3_Audio>(&npu_device_global)};
    }
    if (model_tag == "lfm3-omni" || model_tag == "lfm3-omni:1.7b") {
        return {"lfm3-omni:1.7b", std::make_unique<LFM3_Omni>(&npu_device_global)};
    }
    return {"lfm3:1.7b", std::make_unique<LFM3>(&npu_device_global)};
}



int main(int argc, char* argv[]) {
    #ifdef __WINDOWS__
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    // Set thread priority to low
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    #endif
    
    arg_utils::po::options_description desc("Allowed options");
    arg_utils::po::variables_map vm;
    desc.add_options()("model,m", arg_utils::po::value<std::string>()->required(), "Model file");
    desc.add_options()("Short,s", arg_utils::po::value<bool>()->default_value(true), "Short Prompt");
    desc.add_options()("Preemption,p", arg_utils::po::value<bool>()->default_value(false), "Preemption");
    arg_utils::po::store(arg_utils::po::parse_command_line(argc, argv, desc), vm);

    std::string tag = vm["model"].as<std::string>();
    bool short_prompt = vm["Short"].as<bool>();
    bool preemption = vm["Preemption"].as<bool>();
    std::cout << "Model: " << tag << std::endl;
    std::string exe_dir = utils::get_executable_directory();
    std::string model_dir = utils::get_models_directory();
    std::string model_list_path = exe_dir + "/model_list.json";
    model_list model_list(model_list_path, model_dir);


   
    header_print("info", "Initializing chat model...");
    auto [resolved_tag, chat] = get_lfm3_model(tag);
    std::string model_path = model_list.get_model_path(resolved_tag);
    std::pair<std::string, nlohmann::json> model_info_pair = model_list.get_model_info(resolved_tag);
    nlohmann::json model_info = model_info_pair.second;
    std::cout << "Model path: " << model_path << std::endl;

    npu_device_global = flm_rt::device(0); 
   
    chat->load_model(model_path, model_info, -1, preemption);
    chat->set_topk(1);
    chat_meta_info_t meta_info;
    lm_uniform_input_t uniformed_input;


    if (short_prompt) {
        uniformed_input.prompt = "Who are you? ";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
        uniformed_input.prompt = "What is C. elegans?";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;

        uniformed_input.prompt = "How many cells does it have?";
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        response = chat->generate_with_prompt(meta_info, uniformed_input, 1024, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
    else{
        std::ifstream file("../../../../test.txt", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open prompt file" << std::endl;
            return 1;
        }
        uniformed_input.prompt = "";
        file.seekg(0, std::ios::end);
        uniformed_input.prompt.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(uniformed_input.prompt.data(), uniformed_input.prompt.size());
        file.close();
        std::cout << "Prompt: " << uniformed_input.prompt << std::endl;
        std::cout << "Response: ";
        chat->start_total_timer();
        std::string response = chat->generate_with_prompt(meta_info, uniformed_input, 128, std::cout);
        chat->stop_total_timer();
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << chat->show_profile() << std::endl;
    }
   
    // std::pair<std::string, std::vector<int>> history = chat->get_history();
    // std::cout << "History length: " << history.second.size() << std::endl;

    return 0;
}
