import numpy as np

def select_templates(DATABASE, TYPES):
    """
    Selects templates of a given type(s) from a template database

    Input: DATABASE  list of templates
           TYPES     list of types to be selected

    Output: array of templates of given type(s)
    """

    database_trunc = list([])

    for type in TYPES:
        # Ensure type is a string
        if not isinstance(type, str):
            continue
        # Debug print statement to check types and DATABASE contents
        print(f"Selecting templates for type: {type}")
        selected = [x for x in DATABASE if type in x]
        print(f"Selected templates: {selected}")
        database_trunc += selected

    return np.array(database_trunc)


