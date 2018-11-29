(cd ../lib-potential-field && cargo build --release)
cp ../lib-potential-field/target/release/libpotential_field.a .
matlab -nodisplay -nosplash -nodesktop -r "run('mexify_potential_field.m');exit;"
