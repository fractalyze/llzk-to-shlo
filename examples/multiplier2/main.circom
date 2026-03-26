pragma circom 2.0.0;

// Simple multiplier circuit: out = in1 * in2
template Multiplier2() {
   signal input in1;
   signal input in2;
   signal output out;
   out <== in1 * in2;
}

component main {public [in1,in2]} = Multiplier2();
