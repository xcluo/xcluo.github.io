import numbers
from mkdocs.config import base, config_options as c

class _Rectangle(base.Config):
    width = c.Type(numbers.Real)  # required
    height = c.Type(numbers.Real)  # required

class MyPluginConfig(base.Config):
    add_rectangles = c.ListOfItems(c.SubConfig(_Rectangle))  # required

    def on_pre_build(self, config, **kwargs):
        print('---'*10)
        print('here')
        print('+++'*10)