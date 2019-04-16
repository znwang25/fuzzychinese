import pkg_resources
import logging
default_logger = logging.getLogger(__name__)


class Stroke(object):
    """
    Translate a chinese character into strokes.

    **Parameters**
    ----------

    *dictionary_filepath* : str, default=None. 

    File path for user provided dictionary. Default dictionary will be 
            used if not specified. 

    A valid dictionary should be a "UTF-8" encoded text file, having two 
            columns separated by space. First column is the character and the 
            second column is its corresponding decomposition with each char 
            stands for each stroke. Note, the decomposition does not have to 
            be strokes, it can be numbers or letters, or any sequence of chars 
            you like).

        An example dictionary:

        -------
        上 〡一一
        下 一〡㇔
        -------
    """

    _default_dictionary_filepath = pkg_resources.resource_filename(
        __name__, 'dict_chinese_stroke.txt')

    def __init__(self, dictionary_filepath=None):
        if dictionary_filepath:
            self._dictionary_filepath = dictionary_filepath
        else:
            self._dictionary_filepath = self._default_dictionary_filepath
        self._read_dictionary()

    def _read_dictionary(self):
        self._dictionary = {}
        default_logger.debug('Reading stroke dictionary ...')
        with open(self._dictionary_filepath, encoding="UTF-8") as f:
            for line in f:
                line = line.strip("\n")
                line = line.split(" ")
                self._dictionary[line[0]] = line[1:]

    def get_stroke(self, character, placeholder='', raise_error=False):
        """
        Decompose a character into strokes based on dictionary.

        **Parameters**
        ----------

        *character* : str

        A chinese character to be decomposed.

        *placeholder* : str, default = ''

        Output to be used when a character can not be decomposed.

        *raise_error*: boolean, default = False

        If true, raise error if a character can not be decomposed. 
            The default action is to show warnings.

        **Returns**
        ----------

        str, decomposition results.
        """

        if character in self._dictionary:
            return ''.join(self._dictionary[character])
        else:
            if raise_error:
                raise Exception(
                    f'Unable to decompose character \'{character}\'.')
            else:
                default_logger.warning(
                    f'Unable to decompose character \'{character}\'.')
                return placeholder


if __name__ == "__main__":
    stroke = Stroke()
    print("中", stroke.get_stroke("中"))
    print("王", stroke.get_stroke("王"))
    print("像", stroke.get_stroke("像"))
