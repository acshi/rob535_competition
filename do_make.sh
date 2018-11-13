#!/bin/bash

make all

CPLEXDIR=/opt/ibm/ILOG/CPLEX_Studio128/cplex

cp libcplexqp.a target/debug/deps/
cp libcplexqp.a target/release/deps/
ln -f -s $CPLEXDIR/lib/x86-64_linux/static_pic/libcplex.a target/debug/deps/libcplex.a
ln -f -s $CPLEXDIR/lib/x86-64_linux/static_pic/libcplex.a target/release/deps/libcplex.a
rm target/debug/rob535_competition
