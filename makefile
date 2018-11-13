CPLEXDIR = /opt/ibm/ILOG/CPLEX_Studio128/cplex

all: test_qp libcplexqp.o libcplexqp.a

test_qp: test_qp.c cplex_qp_rust.c
	gcc -o $@ $^ -I $(CPLEXDIR)/include/ -L $(CPLEXDIR)/lib/x86-64_linux/static_pic -lcplex -lpthread -ldl -lm

libcplexqp.o: cplex_qp_rust.c
	gcc -c -fPIC -o $@ $^ -I $(CPLEXDIR)/include/ -L $(CPLEXDIR)/lib/x86-64_linux/static_pic -lcplex -lpthread -ldl -lm

libcplexqp.a: libcplexqp.o
	ar rcs $@ $^
