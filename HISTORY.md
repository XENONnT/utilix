v0.5.3 - 2021/04/14
=====
 * Small bugfixes in utilix.io

v0.5.2 - 2021/04/13
=====
 * Actually fixes the bug introduced in v0.5.0.

v0.5.1 - 2021/04/12
======
 * Fixes the bug introduced in v0.5.0 for cases when no config file exists. (#41, #42)


v0.5.0 - 2021/04/12
======
 * GridFS interface for both direct mongo access and through the API (#35).
 * Allow users to bypass the auto-initialization of the DB class with a config option (#39)
 * Hopefully a fix to the annoying error that requires removing the dbtoken (#40)
 * And a few other small bug fixes / typos.

v0.4.1 - 2021/02/03
======
Minor changes:
  - Don't fail on import (https://github.com/XENONnT/utilix/pull/33)
  - Add version number to utilix, add bumpversion (https://github.com/XENONnT/utilix/pull/31)


v0.4.0 - 2021/01/21
======
Major updates:
- Different collections for XENON1T and XENONnT (#29)

Minor updates
- Add methods for MC collection (#30)
- Bug fixes (#24, #25)
- Windows support (#26)
- Better error handling (#27)
