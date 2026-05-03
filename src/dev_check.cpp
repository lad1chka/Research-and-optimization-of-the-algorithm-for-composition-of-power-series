// Translation unit that pulls in the umbrella header so the library headers
// are compiled once with the project's warning flags, even though pscomp is
// distributed as header-only.

#include "pscomp/pscomp.hpp"
