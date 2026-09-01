from app.services.json_stream import JsonStringFieldAccumulator


def test_json_string_field_accumulator_emits_only_new_decoded_text() -> None:
    accumulator = JsonStringFieldAccumulator("reply")

    assert accumulator.feed('{"rep') == ""
    assert accumulator.feed('ly":"你好\\n') == "你好\n"
    assert accumulator.feed('继续\\u3002') == "继续。"
    assert accumulator.feed('","focus_dimension":"decision"}') == ""
