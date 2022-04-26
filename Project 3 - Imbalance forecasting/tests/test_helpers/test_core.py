from src.helpers.core import list_without_element

def test_list_without_element_removes_correct_element():
    input_list = [-9, -5, 0, 5, 9]
    output_list = list_without_element(input_list, 5)

    assert len(output_list) == len(input_list) - 1
    assert 5 not in output_list
    assert 5 in input_list
