CC=gcc
CXX=g++

LIBNAME=libdmtcp_cuda
LIBOBJS = cuda-ckpt.o
VMM_LIB=libvmm_ckpt.so

DMTCP_INCLUDE_FLAGS=-I$(DMTCP_ROOT)/include -I$(DMTCP_ROOT)/jalib -I$(DMTCP_ROOT)/src

# Find CUDA path
NVCC_PATH := $(shell which nvcc 2>/dev/null)
ifdef NVCC_PATH
  CUDA_HOME := $(realpath $(dir $(NVCC_PATH))/..)
endif
CUDA_HOME ?= $(or $(CUDA_PATH),/usr/local/cuda)
CUDA_INCLUDE_FLAGS=-I$(CUDA_HOME)/include

override CFLAGS   += -g3 -O0 -fPIC $(DMTCP_INCLUDE_FLAGS) $(CUDA_INCLUDE_FLAGS)
override CXXFLAGS += -g3 -O0 -fPIC $(DMTCP_INCLUDE_FLAGS) $(CUDA_INCLUDE_FLAGS)

LINK = ${CXX}

default: ${LIBNAME}.so tests

check: ${LIBNAME}.so tests
	@${DMTCP_ROOT}/bin/dmtcp_command --quit 2>/dev/null || true
	@rm -f ckpt_*.dmtcp dmtcp_restart_script*.sh
	echo Launching ./test/counter
	@${DMTCP_ROOT}/bin/dmtcp_launch --with-plugin $$PWD/${LIBNAME}.so ./test/counter & \
	LAUNCH_PID=$$! ; \
	sleep 3 ; \
	echo Start checkpointing
	${DMTCP_ROOT}/bin/dmtcp_command --checkpoint ; \
	sleep 4 ; \
	${DMTCP_ROOT}/bin/dmtcp_command --quit ; \
	wait $$LAUNCH_PID 2>/dev/null || true ; \
	echo Restarting
	${DMTCP_ROOT}/bin/dmtcp_restart ckpt_*.dmtcp & \
	RESTART_PID=$$! ; \
	sleep 3 ; \
	${DMTCP_ROOT}/bin/dmtcp_command --quit ; \
	wait $$RESTART_PID 2>/dev/null || true
	rm -rf ckpt_*.dmtcp
	rm -rf dmtcp_restart_script*

${LIBNAME}.so: ${LIBOBJS}
	${LINK} -shared -fPIC -o $@ $^ -lcuda -ldl

# VMM proof of concept; does not need DMTCP.
${VMM_LIB}: vmm-ckpt.cpp vmm-ckpt.h
	${CXX} -g3 -O0 -fPIC -shared $(CUDA_INCLUDE_FLAGS) -o $@ vmm-ckpt.cpp \
	  -L$(CUDA_HOME)/lib64 -lcuda -lcudart -ldl

vmm: ${VMM_LIB}
	cd test && $(MAKE) vmm_poc

check-assumptions:
	cd test && $(MAKE) vmm_assumptions
	./test/vmm_assumptions

check-vmm: vmm
	@rm -f vmm-ckpt-*.img
	LD_PRELOAD=$$PWD/${VMM_LIB} ./test/vmm_poc --ckpt-at 3 --image vmm-ckpt-1.img
	LD_PRELOAD=$$PWD/${VMM_LIB} ./test/vmm_poc --restart vmm-ckpt-1.img \
	  --ckpt-at 6 --image vmm-ckpt-2.img
	LD_PRELOAD=$$PWD/${VMM_LIB} ./test/vmm_poc --restart vmm-ckpt-2.img
	@rm -f vmm-ckpt-*.img

.c.o:
	${CC} ${CFLAGS} -c -o $@ $<
.cpp.o:
	${CXX} ${CXXFLAGS} -c -o $@ $<

tests:
	cd test && $(MAKE)

tidy:
	rm -f *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

clean: tidy
	rm -f ${LIBOBJS} ${LIBNAME}.so ${VMM_LIB} vmm-ckpt-*.img
	cd test && $(MAKE) clean

distclean: clean
	rm -f ${LIBNAME}.so *~ .*.swp dmtcp_restart_script*.sh ckpt_*.dmtcp

dist: distclean
	dir=`basename $$PWD`; cd ..; \
	  tar czvf $$dir.tar.gz --exclude-vcs ./$$dir
	dir=`basename $$PWD`; ls -l ../$$dir.tar.gz

.PHONY: default check vmm check-vmm check-assumptions tests tidy clean distclean dist
