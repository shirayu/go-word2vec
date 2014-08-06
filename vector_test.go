package word2vec

import (
	"github.com/shirayu/go-word2vec"
	"testing"
)

func TestVector(t *testing.T) {
	vector1 := word2vec.Vector{2.0, 4.0, 16.0}
	vector2 := word2vec.Vector{1.0, 1.0, 1.0}
	vector3 := word2vec.Vector{2.0, 2.0, 2.0}
	vector4 := word2vec.Vector{5.0, 5.0, 5.0}
	vector5 := word2vec.Vector{7.0, 7.0, 7.0}
	const dot45 = float32(105.0)
	const norm1 = 16.613247

	norm1a := vector1.GetNorm()
	if norm1a != norm1 {
		t.Errorf("Error of GetNorm()\twant: %q\tbut: %q\n", norm1, norm1a)
	}

	norm1b := vector1.Normalize()
	if norm1b != norm1 {
		t.Errorf("Error of Normalize()\twant: %q\tbut: %q\n", norm1, norm1b)
	}

	norm1c := vector1.GetNorm()
	if norm1c != 1 {
		t.Errorf("Error of GetNorm() after Normalize()\twant: %q\tbut: %q\n", 1, norm1c)
	}

	vector2.Scal(2)
	if !vector2.Equals(vector3) {
		t.Errorf("Error of Scal()\twant:%q\tbut: %q", vector3, vector2)
	}

	vector3.Add(1, vector4)
	if !vector3.Equals(vector5) {
		t.Errorf("Error of Add()\twant:%q\tbut: %q", vector5, vector3)
	}

	dot45_sys := vector4.Dot(vector5)
	if dot45_sys != dot45 {
		t.Errorf("Error of Add()\twant:%q\tbut: %q", dot45, dot45_sys)
	}
}
