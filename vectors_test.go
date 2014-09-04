package word2vec

import (
	"testing"
)

func TestVectors(t *testing.T) {
	vector0 := Vector{2.0, 3.0, 1.0}
	vector1 := Vector{2.0, 4.0, 16.0}
	vector2 := Vector{1.0, 1.0, 1.0}
	vector3 := Vector{2.0, 2.0, 2.0}
	vecs := make(Vectors, 4, 4)
	vecs[0] = vector0
	vecs[1] = vector1
	vecs[2] = vector2
	vecs[3] = vector3

	for idx, vec := range vecs {
		c, _ := vecs.HighestDot(vec)
		if c != 1 {
			t.Errorf("Error of GetClosest)\twant:%d\tbut: %d", idx, c)
		}
	}
}
