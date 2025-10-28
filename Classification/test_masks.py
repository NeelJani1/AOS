import os
import sys
sys.path.append('.')

# Temporarily modify to only test 10%
original_import = __import__

def custom_import(name, *args, **kwargs):
    module = original_import(name, *args, **kwargs)
    if name == 'generate_masks':
        # Modify the forget_percentages to only test 10%
        module.forget_percentages = [0.1]
    return module

__builtins__.__import__ = custom_import

from generate_masks import main
main()
