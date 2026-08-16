# accelerInt radau2a built at single or double precision, without SCons.
#
#   make PROBLEM=ring_modulator PRECISION=single RTOL=1e-4 ATOL=1e-6
#   ./build/radau2a-ring_modulator-single <num_threads> <num_IVPs>
#
# ATOL and RTOL are compile-time constants, so a tolerance sweep is one build per point.

ACCELERINT ?= accelerInt-src
BUILD      ?= build
PROBLEM    ?= lorenz
PRECISION  ?= single
PYTHON     ?= python3

ifeq ($(PROBLEM),lorenz)
  PROBLEM_DEF := PROBLEM_LORENZ
  DURATION    := 1.0
  JAC         ?= analytic
else ifeq ($(PROBLEM),ring_modulator)
  PROBLEM_DEF := PROBLEM_RING_MODULATOR
  DURATION    := 1.0e-3
  JAC         ?= fd
else
  $(error unknown PROBLEM '$(PROBLEM)'; expected lorenz or ring_modulator)
endif

ifeq ($(PRECISION),single)
  PREC_DEF  := -DACC_SINGLE
  EPS_MACRO := FLT_EPSILON
  MIN_MACRO := FLT_MIN
  SUFFIX    := f
else ifeq ($(PRECISION),double)
  PREC_DEF  := -DACC_DOUBLE
  EPS_MACRO := DBL_EPSILON
  MIN_MACRO := DBL_MIN
  SUFFIX    :=
else
  $(error unknown PRECISION '$(PRECISION)'; expected single or double)
endif

RTOL     ?= 1e-4
ATOL     ?= 1e-6
END_TIME ?= $(DURATION)
T_STEP   ?= $(DURATION)
N_RA     ?= 10

CAST    := $(BUILD)/accelerInt-$(PRECISION)
GENERIC := $(CAST)/generic
RADAU   := $(CAST)/radau2a
OBJDIR  := $(BUILD)/$(PROBLEM)-$(PRECISION)-$(JAC)$(if $(filter 1,$(LOG)),-log)
TARGET  := $(BUILD)/radau2a-$(PROBLEM)-$(PRECISION)$(if $(filter 1,$(LOG)),-log)

CC      ?= gcc
# gnu99: generic/timer.h uses timersub(). RADAU2A: selects radau2a_props.h, which defines STRIDE.
CFLAGS  := -O3 -std=gnu99 -fopenmp -DRADAU2A -D$(PROBLEM_DEF) $(PREC_DEF) \
           -I$(OBJDIR) -Isrc -Iproblems -I$(GENERIC) -I$(RADAU)
LDFLAGS := -fopenmp
LDLIBS  := -llapack -lblas -lm

# LOG=1 writes the initial and final states to log/radau2a-int-log.bin.
ifeq ($(LOG),1)
  CFLAGS += -DLOG_OUTPUT -DLOG_END_ONLY
endif

# Exactly one of jacob.c and fd_jacob.c is linked; both define eval_jacob.
ifeq ($(JAC),fd)
  JAC_SRC := $(GENERIC)/fd_jacob.c
else
  JAC_SRC := src/jacob.c
endif

# src/status.c replaces radau2a_props.c, whose check_error exits on the first
# failing trajectory.
LOCAL_SRC := src/dydt.c src/ics.c src/sparse_multiplier.c src/status.c $(JAC_SRC)
CAST_SRC  := $(GENERIC)/solver_main.c $(GENERIC)/solver_generic.c \
             $(GENERIC)/solver_interface.c $(GENERIC)/complexInverse.c \
             $(RADAU)/radau2a.c $(RADAU)/radau2a_init.c

OBJ := $(addprefix $(OBJDIR)/,$(notdir $(LOCAL_SRC:.c=.o) $(CAST_SRC:.c=.o)))
VPATH := src:problems:$(GENERIC):$(RADAU)

.PHONY: all build clean distclean cast check

# Two phases: VPATH is scanned at startup, so the cast has to finish before the
# recipe that compiles its output is parsed.
all:
	@$(MAKE) --no-print-directory cast
	@$(MAKE) --no-print-directory build

build: $(TARGET)

$(ACCELERINT)/SConstruct:
	git clone --depth 1 https://github.com/SLACKHA/accelerInt.git $(ACCELERINT)

cast: $(CAST)/.stamp

$(CAST)/.stamp: cast_float32.py $(ACCELERINT)/SConstruct
	@mkdir -p $(CAST)
	$(PYTHON) cast_float32.py $(ACCELERINT) $(CAST) $(PRECISION)
	@touch $@

# Stands in for the header SCons generates.
$(OBJDIR)/solver_options.h:
	@mkdir -p $(OBJDIR)
	@printf '%s\n' \
	  '#ifndef SOLV_OPT_HEAD' \
	  '#define SOLV_OPT_HEAD' \
	  '#include <float.h>' \
	  '#define ATOL ($(ATOL)$(SUFFIX))' \
	  '#define RTOL ($(RTOL)$(SUFFIX))' \
	  '#define t_step ($(T_STEP)$(SUFFIX))' \
	  '#define end_time ($(END_TIME)$(SUFFIX))' \
	  '#define EPS $(EPS_MACRO)' \
	  '#define SMALL $(MIN_MACRO)' \
	  '#define N_RA ($(N_RA))' \
	  '#define SAME_IC' \
	  '#endif' > $@

$(OBJDIR)/%.o: %.c $(OBJDIR)/solver_options.h $(CAST)/.stamp
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

# Checks the analytic Jacobian against a central difference of the RHS.
check: $(BUILD)/jac_check-$(PROBLEM)
	./$<

$(BUILD)/jac_check-$(PROBLEM): tests/jac_check.c src/dydt.c src/jacob.c $(OBJDIR)/solver_options.h
	@mkdir -p $(BUILD)
	$(CC) -O2 -std=gnu99 -DRADAU2A -D$(PROBLEM_DEF) -DACC_DOUBLE \
	  -I$(OBJDIR) -Isrc -Iproblems \
	  tests/jac_check.c src/dydt.c src/jacob.c -lm -o $@

clean:
	rm -rf $(BUILD)/lorenz-* $(BUILD)/ring_modulator-* $(BUILD)/radau2a-* $(BUILD)/jac_check-*

distclean:
	rm -rf $(BUILD) $(ACCELERINT)
