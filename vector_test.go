package word2vec

import (
	"testing"
)

func TestVector(t *testing.T) {
	vector0 := Vector{2.0}
	vector1 := Vector{2.0, 4.0, 16.0}
	vector2 := Vector{1.0, 1.0, 1.0}
	vector3 := Vector{2.0, 2.0, 2.0}
	vector4 := Vector{5.0, 5.0, 5.0}
	vector5 := Vector{7.0, 7.0, 7.0}
	const dot45 = float32(105.0)
	const norm1 = 16.613247

	norm1a := vector1.GetNorm()
	if norm1a != norm1 {
		t.Errorf("Error of GetNorm()\twant: %v\tbut: %v\n", norm1, norm1a)
	}

	norm1b := vector1.Normalize()
	if norm1b != norm1 {
		t.Errorf("Error of Normalize()\twant: %v\tbut: %v\n", norm1, norm1b)
	}

	norm1c := vector1.GetNorm()
	if norm1c != 1 {
		t.Errorf("Error of GetNorm() after Normalize()\twant: %v\tbut: %v\n", 1, norm1c)
	}

	vector2.Scal(2)
	if !vector2.Equals(vector3) {
		t.Errorf("Error of Scal()\twant:%v\tbut: %v", vector3, vector2)
	}

	if vector2.Equals(vector0) {
		t.Errorf("Equals()\tTwo vector length is different. %d vs %d", len(vector0), len(vector2))
	}
	if vector3.Equals(vector4) {
		t.Errorf("Equals()\tTwo vector  is not equal. %v vs %v", vector3, vector4)
	}

	vector3.Add(1, vector4)
	if !vector3.Equals(vector5) {
		t.Errorf("Error of Add()\twant:%v\tbut: %v", vector5, vector3)
	}

	dot45Sys := vector4.Dot(vector5)
	if dot45Sys != dot45 {
		t.Errorf("Error of Add()\twant:%v\tbut: %v", dot45, dot45Sys)
	}
}
