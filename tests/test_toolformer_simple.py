from datacreek.utils import toolformer


def test_insert_tool_calls_basic():
    text = "Call 123 then 456"
    out = toolformer.insert_tool_calls(text, [("num", r"\d+")])
    # only the first match should be replaced
    assert out.count("[TOOL:num(123)]") == 1
    assert "456" in out

    # invalid regex patterns are ignored
    same = toolformer.insert_tool_calls(text, [("bad", "[")])
    assert same == text


def test_execute_tool_calls():
    def shout(arg: str) -> str:
        return arg.upper()

    def fail(_: str) -> str:
        raise ValueError

    txt = "A [TOOL:shout(hi)] B [TOOL:fail(x)] [TOOL:none(y)]"
    tools = {"shout": shout, "fail": fail}
    out = toolformer.execute_tool_calls(txt, tools)
    assert out == "A HI B [TOOL:fail(x)] [TOOL:none(y)]"


def test_generate_with_tools_scored():
    def llm_call(prompt: str) -> str:
        return "calc [TOOL:double(3)]" if "TOOL" in prompt else "fail"

    tools = {"double": lambda a: str(int(a) * 2)}
    score_fn = lambda t: 1 if "6" in t else 0

    res = toolformer.generate_with_tools(
        llm_call,
        "give 3",
        tools,
        insert_patterns=[("double", "3")],
        score_fn=score_fn,
        retries=2,
    )
    assert res == "calc 6"


def test_generate_with_tools_baseline():
    def llm_call(prompt: str) -> str:
        return "prefix [TOOL:shout(word)]" if "TOOL" in prompt else "prefix"

    tools = {"shout": lambda a: a.upper()}

    out = toolformer.generate_with_tools(
        llm_call,
        "say word",
        tools,
        insert_patterns=[("shout", "word")],
        retries=1,
    )
    assert out == "prefix WORD"


def test_generate_with_tools_no_improvement():
    def llm_call(prompt: str) -> str:
        return "bad" if "TOOL" in prompt else "good"

    score_fn = len
    tools = {"noop": lambda a: a}

    out = toolformer.generate_with_tools(
        llm_call,
        "prompt",
        tools,
        insert_patterns=[("noop", "p")],
        score_fn=score_fn,
        retries=1,
    )
    assert out == "good"
