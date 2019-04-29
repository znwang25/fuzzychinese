import pkg_resources
import logging
from ._utils import is_cjk
default_logger = logging.getLogger(__name__)


class Radical(object):
    """
    Translate a chinese character into radicals.

    **Parameters**
    ----------

    *dictionary_filepath* : str, default=None. 

    File path for user provided dictionary. Default dictionary will be 
            used if not specified. 

    A valid dictionary should be a "UTF-8" encoded text file, having two 
            columns separated by space. First column is the character and the 
            second column is its corresponding decomposition with each char 
            stands for each Radical. Note, the decomposition does not have to 
            be radicals, it can be numbers or letters, or any sequence of chars 
            you like).

        An example dictionary:

        -------
        思 田心
        疆 弓土畺
        -------
    """

    _default_dictionary_filepath = pkg_resources.resource_filename(
        __name__, 'dict_chinese_radical.txt')

    def __init__(self, dictionary_filepath=None):
        if dictionary_filepath:
            self._dictionary_filepath = dictionary_filepath
        else:
            self._dictionary_filepath = self._default_dictionary_filepath
        self._read_dictionary()

    def _read_dictionary(self):
        self._dictionary = {}
        data = {}
        default_logger.debug('Reading Radical dictionary ...')
        with open(self._dictionary_filepath, encoding="UTF-8") as f:
            for line in f:
                if (self._dictionary_filepath ==
                        self._default_dictionary_filepath):
                    item_list = line.strip().split('\t')
                    key = item_list[0]
                    value = [i.strip().split() for i in item_list[1:]]
                    self._dictionary[key] = value[0]
                else:
                    item_list = line.strip().split(' ')
                    key = item_list[0]
                    value = item_list[1:]
                    self._dictionary[key] = value

    def get_radical(self, character, placeholder='', raise_error=False):
        """
        Decompose a character into radicals based on dictionary. When a character can not be decomposed, itself will be returned. If it's not chinese, a placeholder is returned.

        **Parameters**
        ----------

        *character* : str

        A chinese character to be decomposed.

        *placeholder* : str, default = ''

        Output to be used when the character is not chinese.

        *raise_error*: boolean, default = False

        If true, raise error if a character can not be decomposed. 
            The default action is to show warnings.

        **Returns**
        ----------

        str, decomposition results.
        """
        if raise_error:
            if not is_cjk(character):
                raise Exception(f'Character \'{character}\' is not Chinese.')
            if character in self._dictionary:
                return ''.join(self._dictionary[character])
            else:
                raise Exception(
                    f'Unable to decompose character \'{character}\'.')
        else:
            if not is_cjk(character):
                default_logger.warning(
                    f'Character \'{character}\' is not Chinese.')
                return placeholder
            if character in self._dictionary:
                return ''.join(self._dictionary[character])
            else:
                default_logger.warning(
                    f'Unable to decompose character \'{character}\'.')
                return character


if __name__ == "__main__":
    radical = Radical()
    print("中", radical.get_radical("中"))
    print("王", radical.get_radical("王"))
    print("像", radical.get_radical("像"))
