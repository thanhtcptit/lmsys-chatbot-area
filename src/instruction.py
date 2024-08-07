INSTRUCTION_TEXT = (
    "Given a prompt and two responses #a and #b, evaluate which response is superior or if both "
    "responses are equally good."
)

PROMPT_TEMPLATE = (
    "<Prompt>:<|reserved_special_token_50|>\n{prompt}\n<|reserved_special_token_51|>"
    "\n\n<Response #a>:\n<|reserved_special_token_52|>\n{resp_a}\n<|reserved_special_token_53|>"
    "\n\n<Response #b>:\n<|reserved_special_token_54|>\n{resp_b}\n<|reserved_special_token_55|>"
    "\n\nEvaluate which response is superior or if both responses are equally good. "
    "Answer with a, b, or tie.\n### Answer:"\
)