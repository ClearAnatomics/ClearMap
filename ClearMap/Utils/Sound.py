# -*- coding: utf-8 -*-
"""
Sound
=====

Module providing some simple sound output to signal processes are done.

Example
-------

>>> import ClearMap.Utils.Sound as snd
>>> snd.beep()
"""
__author__    = 'Christoph Kirst <christoph.kirst.ck@gmail.com>'
__license__   = 'GPLv3 - GNU General Pulic License v3 (see LICENSE)'
__copyright__ = 'Copyright © 2020 by Christoph Kirst'
__webpage__   = 'http://idisco.info'
__download__  = 'http://www.github.com/ChristophKirst/ClearMap2'


import os


def beep(duration = 1, frequency = 340):
  os.system('(speaker-test -t sine -f %d >/dev/null)& pid=$! ; sleep %fs ; kill -9 $pid' % (frequency, duration))
  os.system('echo -e "\a" >/dev/null');
  
 
if __name__ == "__main__":
  import ClearMap.Utils.Sound as snd;
  snd.beep(frequency= 440, duration = 0.5)  