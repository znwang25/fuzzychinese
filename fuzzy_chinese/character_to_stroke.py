import pkg_resources


class Stroke(object):
    _dictionary_filepath = pkg_resources.resource_filename(
        __name__, 'dict_chinese_stroke.txt')

    def __init__(self):
        self._dictionary = {}
        self._read_dictionary()

    def _read_dictionary(self):
        self._dictionary = {}
        with open(self._dictionary_filepath, encoding="UTF-8") as f:
            for line in f:
                line = line.strip("\n")
                line = line.split(" ")
                self._dictionary[line[0]] = line[1:]
        # print(self.dictionary)

    def get_stroke(self, word):
        if word in self._dictionary:
            return ''.join(self._dictionary[word])
        else:
            return ''


if __name__ == "__main__":
    stroke = Stroke()
    print("中", stroke.get_stroke("中"))
    print("王", stroke.get_stroke("王"))
    print("像", stroke.get_stroke("像"))
