package word2vec

import (
	"github.com/ziutek/blas"
)

type Vector []float32

// Get the norm of vector
func (v Vector) GetNorm() float32 {
	return blas.Snrm2(len(v), v, 1)
}

// Normalize vector and return the original norm
func (v Vector) Normalize() float32 {
	w := v.GetNorm()
	blas.Sscal(len(v), 1/w, v, 1)
	return w
}

//y = y+ (alpha * x)
func (y Vector) Add(alpha float32, x Vector) {
	blas.Saxpy(len(y), alpha, x, 1, y, 1)
}

//y = y * alpha
func (y Vector) Scal(alpha float32) {
	blas.Sscal(len(y), alpha, y, 1)
}

//return y dot x
func (y Vector) Dot(x Vector) float32 {
	return blas.Sdot(len(y), x, 1, y, 1)
}

// Check whether this equals the given vector
func (y Vector) Equals(x Vector) bool {
	if len(y) != len(x) {
		return false
	}
	for i, val := range x {
		if val != y[i] {
			return false
		}
	}
	return true
}
