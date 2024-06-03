#[derive(Clone, Copy, Debug)]
pub enum Vocab {
    MistralV3, // inst fn_call
    Llama3,    // inst ~fn_call roles
    Phi3,      // inst fn_call fim
}

use Vocab::*;

impl Vocab {
    // mistral: must be followed by { content }</s>
    pub fn user(self) -> [&'static str; 2] {
        match self {
            MistralV3 => ["[INST]", "[/INST]"],
            Llama3 => ["<|start_header_id|>user<|end_header_id|>\n", "<|eot_id|>"],
            Phi3 => ["\n<|user|>\n", "<|end|>"],
        }
    }

    pub fn assistant(self) -> [&'static str; 2] {
        match self {
            MistralV3 => ["", "</s>"],
            Llama3 => [
                "<|start_header_id|>assistant<|end_header_id|>\n",
                "<|eot_id|>",
            ],
            Phi3 => ["\n<|assistant|>\n", "<|end|>"],
        }
    }

    // mistral: must be followed by { content }</s>
    pub fn system(self) -> [&'static str; 2] {
        match self {
            MistralV3 => ["[INST] [SYS]", "[/SYS] [/INST]"],
            Llama3 => ["<|start_header_id|>system<|end_header_id|>\n", "<|eot_id|>"],
            Phi3 => ["\n<|system|>\n", "<|end|>"],
        }
    }

    pub fn tool_list(self) -> [&'static str; 2] {
        match self {
            MistralV3 => ["[AVAILABLE_TOOLS]", "[/AVAILABLE_TOOLS]"],
            Llama3 => [
                "<|start_header_id|>available_tools<|end_header_id|>\n",
                "<|eot_id|>",
            ],
            Phi3 => ["\n<|function_list|>\n", "<|end|>"],
        }
    }

    pub fn tool_out(self) -> [&'static str; 2] {
        match self {
            MistralV3 => ["[TOOL_RESULTS]", "[/TOOL_RESULTS]"],
            Llama3 => [
                "<|start_header_id|>tool_results<|end_header_id|>\n",
                "<|eot_id|>",
            ],
            Phi3 => ["\n<|function_output|>\n", "<|end|>"],
        }
    }
}
