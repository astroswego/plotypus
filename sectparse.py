"""A quick iterator for dividing sections denoted by a heading

By Dan Wysocki <dwysocki@oswego.edu>

Simple usage example:

>>> text = '''Section 1
... Some text
... 
... Section 2
... Some more text
... '''
>>> textit = iter(text.splitlines())
>>> keylist = ['Section 1', 'Section 2']
>>> parser = SectionParser(textit, keylist)
>>> print({heading: text for heading, text in parser})
{None: '', 'Section 1': 'Some text\n', 'Section 2': 'Some more text'}
"""

__version__ = "0.1.1"

__all__ = ["SectionParser",
           "MissingSectionError",
           "DuplicateKeyError"]

__copyright__ = """
       DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
"""

def _hasDuplicates(lst):
    st = set(lst)
    return len(st) != len(lst)

def _repr(self):
    return "<%s at 0x%x>" % (self.__class__.__name__, id(self))

class DuplicateKeyError(Exception):
    """Raised if there are duplicate keys in the keylist"""

    def __init__(self, lstname):
        self.lstname = lstname
    def __str__():
        return "Duplicate keys found in %s" % self.lstname

class MissingSectionError(Exception):
    """Raised if a key in keylist does not have a section by the end of input

    """
    
    def __init__(self, keylist):
        self.keylist = keylist

    def __str__(self):
        return "Keys missing from input: {}".format(
            ", ".join(str(key) for key in self.keylist))


class SectionParser:
    """Takes two parameters
      input   -- An iterable containing the text to be parsed.
      keylist -- A key listing section headings to search for. If any are
                 missing when the end of the input is reached, a MissingSectionError
                 is raised, and if any are duplicates, a DuplicateKeyError is
                 raised.

    Each iteration returns the 2-tuple (<section heading>, <section text>)

    """
    def __init__(self, input, keylist):
        # Makes sure the input is an iterator and not a sequence
        self.__input = iter(input)
        # Raises exception if there are duplicate keys
        if _hasDuplicates(keylist):
            raise DuplicateKeyError("keylist")
        # List of section headings
        self.__keylist = keylist
        # The heading for the next section
        self.__nextkey = None

    __repr__ = _repr
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # If __nextkey is defined, reads up until the key after it, or until
        # the end of the input, and returns (section heading, section contents)
        if self.__nextkey or self.__keylist:
            self.__nextkey, lastkey, text = self.__nextSection()
            if self.__nextkey or lastkey:
                return lastkey, text
            else:
                raise (MissingSectionError(self.__keylist) if self.__keylist
                                                           else StopIteration)
        else:
            # Out of keys and end of input
            raise StopIteration
    
    def __nextSection(self):
        """Returns the next key, the current key, the section's contents, and
        the rest of the input.
        
        """
        nextKey = None
        sectionKey = self.__nextkey
        sectionText = ""
        for line in self.__input:
            line = line.replace("\n", "") # Removes trailing newline
            # If the line is the next key, it is removed from __keylist and set
            # as the nextKey to be returned, then the function returns,
            # otherwise the line is appended to sectionText
            if line in self.__keylist:
                nextKey = line
                self.__keylist.remove(nextKey)
                break
            else:
                sectionText += ("\n" if sectionText else "") + line

        return nextKey, sectionKey, sectionText

