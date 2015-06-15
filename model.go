package word2vec

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

type Model struct {
	vocabSize  int
	vectorSize int
	vocab      map[string]int
	data       Vector //connect all vectors
	norms      Vector //connect all vectors
}

func NewModel(inf *os.File) (model *Model, err error) {
	model = new(Model)
	reader := bufio.NewReader(inf)
	fmt.Fscanln(reader, &model.vocabSize, &model.vectorSize)

	model.vocab = make(map[string]int)
	model.data = make(Vector, model.vectorSize*model.vocabSize)
	model.norms = make(Vector, model.vocabSize)

	tmp := make(Vector, model.vectorSize)
	var myword string
	for i := 0; i < model.vocabSize; i++ {
		bytes, err := reader.ReadBytes(' ')
		if err != nil {
			return model, err
		}
		myword = string(bytes[1 : len(bytes)-1])

		model.vocab[myword] = i
		binary.Read(reader, binary.LittleEndian, tmp)

		originalNorm := tmp.Normalize()
		model.norms[i] = originalNorm
		copy(model.data[i*model.vectorSize:(i+1)*model.vectorSize], tmp)
	}

	return model, nil
}
func (self *Model) GetVocab() map[string]int {
	return self.vocab
}
func (self *Model) GetVocabSize() int {
	return self.vocabSize
}
func (self *Model) GetConnectedVector() Vector {
	return self.data
}
func (self *Model) GetVectorSize() int {
	return self.vectorSize
}

//Get normalized vector and its original norm
func (self *Model) GetVector(word string) (Vector, float32) {
	position, ok := self.vocab[word]
	if ok {
		norm := self.norms[position]
		return self.data[position*self.vectorSize : (position+1)*self.vectorSize], norm
	}
	return nil, 0
}

//Get normalized vector
func (self *Model) GetNormalizedVector(word string) Vector {
	position, ok := self.vocab[word]
	if ok {
		return self.data[position*self.vectorSize : (position+1)*self.vectorSize]
	}
	return nil
}

//Get original norm
func (self *Model) GetNorm(word string) float32 {
	position, ok := self.vocab[word]
	if ok {
		return self.norms[position]
	}
	return 0
}

//Get similarity of two given words
func (self *Model) Similarity(word1, word2 string) (float32, error) {
	vector1 := self.GetNormalizedVector(word1)
	if vector1 == nil {
		return 0, errors.New(word1 + " not found")
	}
	vector2 := self.GetNormalizedVector(word2)
	if vector2 == nil {
		return 0, errors.New(word2 + " not found")
	}

	return vector1.Dot(vector2), nil
}
