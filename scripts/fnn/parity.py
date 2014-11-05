# parity.py
#
# calculate the parity function for a given input integer
# Daniel E. Worrall -- 16 Oct 2014

# The script computes the number of ones N in the binary representation
# of an integer Z in O(log2(Z)) operations

"""Calculate the parity function for a given input integer"""

class Parity:
    def __call__(self):
        pass

    # temporarily generate real valued, full rank, square matrices
    def parity(self,Z):
        # Z     the integer argument of the parity function
        
        self.par_count = 0
        while Z:
            self.par_count += Z%2
            Z /= 2
            
        return self.par_count %2
		






















