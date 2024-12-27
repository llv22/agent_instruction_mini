import re
import tokenize
from io import StringIO

def replace_action(a):
    ra = re.sub(r"\\+n", "\n", a)
    ra = re.sub(r"\\+t", "\t", ra)
    ra = re.sub(r"\\+(?![ntrbfvauxo\'\"])", "", ra)
    return ra

def escape_unbalanced_quotes(a):
    """Escape unbalanced single or double quotes."""
    single_quote_count = a.count("'")
    double_quote_count = a.count('"')

    # Escape last unmatched quotes if unbalanced
    if single_quote_count % 2 != 0:
        a = a.replace("'", "\\'", 1)
    if double_quote_count % 2 != 0:
        a = a.replace('"', '\\"', 1)
    return a

def remove_comments_with_tokenize(a):
    """Remove comments from a Python code string using the tokenize module.

    Args:
        a (str): original action string in python code format

    Returns:
        str: python code with comments removed
    """
    # see: heuristics to replace \\+n with \n
    # ra = replace_action(a)
    # ra = escape_unbalanced_quotes(ra)
    tokens = tokenize.generate_tokens(StringIO(a).readline)
    result = []
    for token in tokens:
        if token.type not in (tokenize.COMMENT, tokenize.NL):
            result.append(token.string)
    return replace_action("".join(result))
    # return "".join(result)

if __name__ == "__main__":
    # action = "click('768')'click('804')"
    # parsed_action = remove_comments_with_tokenize(action)
    # print(parsed_action)
    command = 'send_msg_to_user("The top 3 best-selling products in January 2023 are:\\\n1. Quest Lumaflex™ Band - $19.00 (Quantity: 6)\\\n2. Sprite Stasis Ball 65 cm - $27.00 (Quantity: 6)\\\n3. Overnight Duffle - $45.00 (Quantity: 5)")'
    parsed_action = remove_comments_with_tokenize(command)
    print(parsed_action)
