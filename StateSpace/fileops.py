#Created by Breschine Cummins on June 20, 2012.

# Copyright (C) 2012 Breschine Cummins
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 59 Temple
# Place, Suite 330, Boston, MA 02111-1307 USA
#

import cPickle, os

def makePath(wholepath,basedir,basename):
    basedir = os.path.dirname(wholepath) or basedir
    basename = os.path.basename(wholepath) or basename
    if not basename.endswith('.pickle'):
        basename = basename+'.pickle'
    return os.path.join(basedir,basename)

def dumpPickle(mydict,wholepath='',basedir='',basename=''):
    '''
    Pickles a dictionary into a file name. File name is given either
    with the full path in wholepath, or with the path split into two 
    pieces such that wholepath = basedir/basename. If wholepath is 
    specified, then basedir and basename are ignored. Program appends 
    .pickle to the file name if not present.

    '''
    wholepath = makePath(wholepath,basedir,basename)
    F = open(wholepath, 'w')
    cPickle.Pickler(F).dump(mydict)
    F.close()

def loadPickle(wholepath='',basedir='',basename=''):
    '''
    Returns an unpickled dictionary from a file name specified either 
    with the full path in wholepath, or with the path split into two 
    pieces such that wholepath = basedir/basename. If wholepath is 
    specified, then basedir and basename are ignored. Program appends 
    .pickle to the file name if not present.

    '''
    wholepath = makePath(wholepath,basedir,basename)
    F = open(wholepath, 'r')
    mydict = cPickle.Unpickler(F).load()
    F.close()
    return mydict
