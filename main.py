from commongen_validation_test_set_generation import language_models


def result_separator(output: str, content_words: list[str]) -> str:
    return output.split(sep=f'Create a sentence with the following words: {", ".join(content_words)}.')[-1].strip(
        ' \t\n\r')


if __name__ == '__main__':
    # checkpoint = "bigscience/bloomz-3b"
    #
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    #
    # inputs = tokenizer(
    #     "Create a sentence with the following words: ski, mountain, fall.",
    #     return_tensors="pt"
    # ).to("cuda")
    # # def tokenizer_wrapper(*args, **kwargs)
    # outputs = model.generate(
    #     **inputs,
    #     num_beams=4,
    #     num_beam_groups=4,
    #     num_return_sequences=4,
    #     diversity_penalty=100.0,
    #     remove_invalid_values=True,
    #     temperature=1.0,
    #     max_new_tokens=256,
    #     return_dict_in_generate=True,
    #     output_scores=True,
    # )
    # for output in outputs.sequences:
    #     sentence = tokenizer.decode(output, skip_special_tokens=True)
    #     print(result_separator(sentence, ["ski", "mountain", "fall"]))
    template = '''
                                <tr class="para-row">
                                <td><p>${LLM_NAME}</p></td>
                                <td>
                                    <div>
                                        <label><input type="radio" name="LLM_NAME-fluency" value=1>1</label>
                                        <label><input type="radio" name="LLM_NAME-fluency" value=2>2</label>
                                        <label><input type="radio" name="LLM_NAME-fluency" value=3>3</label>
                                        <label><input type="radio" name="LLM_NAME-fluency" value=4>4</label>
                                        <label><input type="radio" name="LLM_NAME-fluency" value=5>5</label>
                                    </div>
                                </td>
                                <td>
                                    <div>
                                        <label><input type="radio" name="LLM_NAME-sensibility" value=1>1</label>
                                        <label><input type="radio" name="LLM_NAME-sensibility" value=2>2</label>
                                        <label><input type="radio" name="LLM_NAME-sensibility" value=3>3</label>
                                        <label><input type="radio" name="LLM_NAME-sensibility" value=4>4</label>
                                        <label><input type="radio" name="LLM_NAME-sensibility" value=5>5</label>
                                    </div>
                                </td>
                                <td>
                                    <div>
                                        <label><input type="radio" name="LLM_NAME-complexity" value=1>1</label>
                                        <label><input type="radio" name="LLM_NAME-complexity" value=2>2</label>
                                        <label><input type="radio" name="LLM_NAME-complexity" value=3>3</label>
                                        <label><input type="radio" name="LLM_NAME-complexity" value=4>4</label>
                                        <label><input type="radio" name="LLM_NAME-complexity" value=5>5</label>
                                    </div>
                                </td>
                            </tr>
    '''
    result = ""
    for language_model in language_models.get_language_models():
        filled_template = template.replace("LLM_NAME", language_model)
        result += filled_template
    print(result)
