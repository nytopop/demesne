use llama_sys::{Token, Tokenizer};
use std::str::FromStr;

#[derive(Clone, Copy, Debug)]
pub enum TemplateKind {
    MistralV3, // inst fn_call
    Llama3,    // inst ~fn_call roles
    Phi3,      // inst fn_call fim
    ChatML,    // ?
}

use TemplateKind::*;

impl FromStr for TemplateKind {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mis3" => Ok(MistralV3),
            "l3" => Ok(Llama3),
            "phi3" => Ok(Phi3),
            "cml" => Ok(ChatML),

            _ => Err("expected 'mis3' | 'l3' | 'phi3' | 'cml'"),
        }
    }
}

#[derive(Default)]
pub struct Template {
    // system
    sys_prefix: Vec<Token>,
    sys_suffix: Vec<Token>,

    // user
    user_prefix: Vec<Token>,
    user_suffix: Vec<Token>,

    // assistant &| tool call
    gen_prefix: Vec<Token>,
    gen_choice_main: Vec<Token>,
    gen_choice_tool: Vec<Token>,
    gen_midfix: Vec<Token>,
    gen_suffix: Vec<Token>,

    // tool list
    list_prefix: Vec<Token>,
    list_suffix: Vec<Token>,

    // tool result
    res_prefix: Vec<Token>,
    res_suffix: Vec<Token>,
}

impl Template {
    pub fn compile(tk: &Tokenizer, kind: TemplateKind) -> Self {
        let mut tmpl = Self::default();

        match kind {
            MistralV3 => {
                // system
                tk.tokenize(&mut tmpl.sys_prefix, "[INST] <<<SYS>>>\n", true);
                tk.tokenize(&mut tmpl.sys_suffix, "<<</SYS>>> [/INST]", true);

                // user
                tk.tokenize(&mut tmpl.user_prefix, "[INST]", true);
                tk.tokenize(&mut tmpl.user_suffix, "[/INST]", true);

                // gen
                tk.tokenize(&mut tmpl.gen_choice_tool, "[TOOL_CALLS]", true);
                tk.tokenize(&mut tmpl.gen_suffix, "</s>", true);

                // list
                tk.tokenize(&mut tmpl.list_prefix, "[AVAILABLE_TOOLS]", true);
                tk.tokenize(&mut tmpl.list_suffix, "[/AVAILABLE_TOOLS]", true);

                // result
                tk.tokenize(&mut tmpl.res_prefix, "[TOOL_RESULTS]", true);
                tk.tokenize(&mut tmpl.res_suffix, "[/TOOL_RESULTS]", true);
            }

            Llama3 => {
                // system
                let s = "<|start_header_id|>system<|end_header_id|>\n";
                tk.tokenize(&mut tmpl.sys_prefix, s, true);
                tk.tokenize(&mut tmpl.sys_suffix, "<|eot_id|>", true);

                // user
                let s = "<|start_header_id|>user<|end_header_id|>\n";
                tk.tokenize(&mut tmpl.user_prefix, s, true);
                tk.tokenize(&mut tmpl.user_suffix, "<|eot_id|>", true);

                // gen
                tk.tokenize(&mut tmpl.gen_prefix, "<|start_header_id|>", true);
                tk.tokenize(&mut tmpl.gen_choice_main, "assistant", true);
                tk.tokenize(&mut tmpl.gen_choice_tool, "calls", true);
                tk.tokenize(&mut tmpl.gen_midfix, "<|end_header_id|>\n", true);
                tk.tokenize(&mut tmpl.gen_suffix, "<|eot_id|>", true);

                // list
                let s = "<|start_header_id|>available_tools<|end_header_id|>\n";
                tk.tokenize(&mut tmpl.list_prefix, s, true);
                tk.tokenize(&mut tmpl.list_suffix, "<|eot_id|>", true);

                // result
                let s = "<|start_header_id|>tool_results<|end_header_id|>\n";
                tk.tokenize(&mut tmpl.res_prefix, s, true);
                tk.tokenize(&mut tmpl.res_suffix, "<|eot_id|>", true);
            }

            Phi3 => {
                // system
                tk.tokenize(&mut tmpl.sys_prefix, "\n<|system|>\n", true);
                tk.tokenize(&mut tmpl.sys_suffix, "<|end|>", true);

                // user
                tk.tokenize(&mut tmpl.user_prefix, "\n<|user|>\n", true);
                tk.tokenize(&mut tmpl.user_suffix, "<|end|>", true);

                // gen
                tk.tokenize(&mut tmpl.gen_prefix, "\n", true);
                tk.tokenize(&mut tmpl.gen_choice_main, "<|assistant|>", true);
                tk.tokenize(&mut tmpl.gen_choice_tool, "<|function_call|>", true);
                tk.tokenize(&mut tmpl.gen_midfix, "\n", true);
                tk.tokenize(&mut tmpl.gen_suffix, "<|end|>", true);

                // list
                tk.tokenize(&mut tmpl.list_prefix, "\n<|function_list|>\n", true);
                tk.tokenize(&mut tmpl.list_suffix, "<|end|>", true);

                // result
                tk.tokenize(&mut tmpl.res_prefix, "\n<|function_output|>\n", true);
                tk.tokenize(&mut tmpl.res_suffix, "<|end|>", true);
            }

            ChatML => {
                // system
                tk.tokenize(&mut tmpl.sys_prefix, "<|im_start|>system\n", true);
                tk.tokenize(&mut tmpl.sys_suffix, "<|im_end|>\n", true);

                // user
                tk.tokenize(&mut tmpl.user_prefix, "<|im_start|>user\n", true);
                tk.tokenize(&mut tmpl.user_suffix, "<|im_end|>\n", true);

                // gen
                tk.tokenize(&mut tmpl.gen_prefix, "<|im_start|>", true);
                tk.tokenize(&mut tmpl.gen_choice_main, "assistant", true);
                tk.tokenize(&mut tmpl.gen_choice_tool, "calls", true);
                tk.tokenize(&mut tmpl.gen_midfix, "\n", true);
                tk.tokenize(&mut tmpl.gen_suffix, "<|im_end|>\n", true);

                // list
                tk.tokenize(&mut tmpl.list_prefix, "<|im_start|>available_tools\n", true);
                tk.tokenize(&mut tmpl.list_suffix, "<|im_end|>\n", true);

                // result
                tk.tokenize(&mut tmpl.res_prefix, "<|im_start|>tool_results\n", true);
                tk.tokenize(&mut tmpl.res_suffix, "<|im_end|>\n", true);
            }
        }

        tmpl
    }

    pub fn feed_sys(&self, tk: &Tokenizer, buf: &mut Vec<Token>, text: &str) {
        buf.extend_from_slice(&self.sys_prefix);
        tk.tokenize(buf, text, false);
        buf.extend_from_slice(&self.sys_suffix);
    }

    pub fn feed_user(&self, tk: &Tokenizer, buf: &mut Vec<Token>, text: &str) {
        buf.extend_from_slice(&self.user_prefix);
        tk.tokenize(buf, text, false);
        buf.extend_from_slice(&self.user_suffix);
    }

    pub fn feed_gen_main(&self, tk: &Tokenizer, buf: &mut Vec<Token>, text: &str) {
        self.prep_gen_choice(buf);
        self.prep_gen_main(buf);
        tk.tokenize(buf, text, false);
        buf.extend_from_slice(&self.gen_suffix);
    }

    pub fn feed_gen_tool(&self, tk: &Tokenizer, buf: &mut Vec<Token>, text: &str) {
        self.prep_gen_choice(buf);
        self.prep_gen_tool(buf);
        tk.tokenize(buf, text, false);
        buf.extend_from_slice(&self.gen_suffix);
    }

    pub fn prep_gen_choice(&self, buf: &mut Vec<Token>) {
        buf.extend_from_slice(&self.gen_prefix);
    }

    pub fn prep_gen_main(&self, buf: &mut Vec<Token>) {
        buf.extend_from_slice(&self.gen_choice_main);
        buf.extend_from_slice(&self.gen_midfix);
    }

    pub fn prep_gen_tool(&self, buf: &mut Vec<Token>) {
        buf.extend_from_slice(&self.gen_choice_tool);
        buf.extend_from_slice(&self.gen_midfix);
    }

    pub fn feed_list(&self, tk: &Tokenizer, buf: &mut Vec<Token>, text: &str) {
        buf.extend_from_slice(&self.list_prefix);
        tk.tokenize(buf, text, false);
        buf.extend_from_slice(&self.list_suffix);
    }

    pub fn feed_result(&self, tk: &Tokenizer, buf: &mut Vec<Token>, text: &str) {
        buf.extend_from_slice(&self.res_prefix);
        tk.tokenize(buf, text, false);
        buf.extend_from_slice(&self.res_suffix);
    }
}
