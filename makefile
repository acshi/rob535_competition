CPLEXDIR = /opt/ibm/ILOG/CPLEX_Studio128/cplex

all: test_qp libcplexrust.o libcplexrust.a

test_qp: test_qp.c src/cplexrust.c
	gcc -g -o $@ $^ -I $(CPLEXDIR)/include/ -L $(CPLEXDIR)/lib/x86-64_linux/static_pic -lcplex -lpthread -ldl -lm

libcplexrust.o: src/cplexrust.c
	gcc -g -c -fPIC -o $@ $^ -I $(CPLEXDIR)/include/

libcplexrust.a: libcplexrust.o
	ar rcs $@ $^

clean:
	rm test_qp libcplexrust.o libcplexrust.a
