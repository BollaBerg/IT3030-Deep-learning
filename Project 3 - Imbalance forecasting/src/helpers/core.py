def list_without_element(lst: list, element) -> list:
    output = lst.copy()
    output.remove(element)
    return output

def format_steps(step: int) -> str:
    minutes = step * 5
    hours = minutes // 60
    minutes -= 60 * hours
    
    if hours == 1:
        return f"{hours} hour, {minutes} minutes"
    elif hours > 1:
        return f"{hours} hours, {minutes} minutes"
    else:
        return f"{minutes} minutes"
