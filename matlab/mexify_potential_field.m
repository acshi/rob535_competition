mex -g -ldl potential_field.c ../lib-potential-field/target/debug/libpotential_field.a
mex -g potential_field.c ../lib-potential-field/target/debug/potential_field.lib WS2_32.Lib Userenv.lib
