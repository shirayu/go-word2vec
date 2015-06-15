package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"github.com/shirayu/go-word2vec"
	"io"
	"os"
	"strings"
)

func getFile(ifname string, ofname string) (inf, outf *os.File, err error) {

	if ifname == "-" {
		inf = os.Stdin
	} else {
		inf, err = os.Open(ifname)
		if err != nil {
			return nil, nil, err
		}
	}

	if ofname == "-" {
		outf = os.Stdout
	} else {
		outf, err = os.Create(ofname)
		if err != nil {
			return inf, nil, err
		}
	}

	return inf, outf, nil
}

func getW2VModel(ifname string) (model *word2vec.Model, err error) {
	if ifname == "" {
		return nil, errors.New("Word2Vec file name is not given")
	}
	w2f, err := os.Open(ifname)
	//     defer w2f.Close()
	if err != nil {
		return nil, err
	}
	return word2vec.NewModel(w2f)
}

func outSims(model *word2vec.Model, vec1 word2vec.Vector, outf *os.File, top_n int) {
	best_words := make([]string, top_n)
	best_vals := make([]float32, top_n)
	vocab := model.GetVocab()
	connected_vector := model.GetConnectedVector()
	vectorSize := model.GetVectorSize()

	for i, _ := range best_vals {
		best_vals[i] = -1
	}

	for word, position := range vocab {
		vec2 := connected_vector[position*vectorSize : (position+1)*vectorSize]
		val := vec1.Dot(vec2)
		for idx := top_n - 1; idx >= 0; idx-- {
			myval := best_vals[idx]
			if val > myval {
				for idx2 := 0; idx2 < idx; idx2++ {
					best_vals[idx2] = best_vals[idx2+1]
					best_words[idx2] = best_words[idx2+1]
				}
				best_vals[idx] = val
				best_words[idx] = word
				break
			}
		}
	}

	for i := top_n - 1; i >= 0; i-- {
		word := best_words[i]
		fmt.Fprintf(outf, "%s %f\n", word, best_vals[i])
	}
	fmt.Fprintf(outf, "\n")
}

func getConnectedVector(model *word2vec.Model, line string) (word2vec.Vector, error) {
	vectorSize := model.GetVectorSize()

	vec1 := make(word2vec.Vector, vectorSize)
	words := strings.Split(line, "|")
	for _, w := range words {
		vec, norm := model.GetVector(w)
		if vec == nil {
			return nil, errors.New("out of vocablary")
		}
		vec1.Add(norm, vec)
	}
	vec1.Normalize()
	return vec1, nil
}

func main() {
	var (
		ifname  string
		ofname  string
		w2vname string
		top_n   int
	)

	f := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	f.StringVar(&ifname, "i", "-", "Input file name. - or no designation means STDIN")
	f.StringVar(&ofname, "o", "-", "Output file name. - or no designation means STDOUT")
	f.IntVar(&top_n, "n", 10, "Top n to show")
	f.StringVar(&w2vname, "m", "", "Word2Vec model file")

	f.Parse(os.Args[1:])
	for 0 < f.NArg() {
		f.Parse(f.Args()[1:])
	}

	inf, outf, err := getFile(ifname, ofname)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}

	model, err := getW2VModel(w2vname)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "Vocabsize: %d\n", model.GetVocabSize())
	fmt.Fprintf(os.Stderr, "Vectorsize: %d\n", model.GetVectorSize())

	reader := bufio.NewReader(inf)
	line, _, err := reader.ReadLine()
	for ; ; line, _, err = reader.ReadLine() {
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "%q\n", err)
			os.Exit(1)
		}

		items := strings.Fields(string(line))
		if len(items) < 2 { // non target line
			if len(line) != 0 {
				x := string(line)
				vec1, err := getConnectedVector(model, x)
				if err == nil {
					outSims(model, vec1, outf, top_n)
				} else {
					fmt.Fprintf(os.Stderr, "%q\n", err)
				}
			}
		} else {
			x := items[0]
			y := items[1]
			vec1, err := getConnectedVector(model, x)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %q\n", x, err)
				continue
			}
			vec2, err := getConnectedVector(model, y)
			if err != nil {
				fmt.Fprintf(os.Stderr, "%s: %q\n", y, err)
				continue
			}
			simval := vec1.Dot(vec2)
			fmt.Fprintf(outf, "%s\t%s\t%f\n", x, y, simval)
		}
	}

}
