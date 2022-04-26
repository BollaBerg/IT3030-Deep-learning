def list_without_element(lst: list, element) -> list:
    output = lst.copy()
    output.remove(element)
    return output