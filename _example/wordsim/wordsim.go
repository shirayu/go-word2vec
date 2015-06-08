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

func main() {
	var (
		ifname  string
		ofname  string
		w2vname string
	)

	f := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	f.StringVar(&ifname, "i", "-", "Input file name. - or no designation means STDIN")
	f.StringVar(&ofname, "o", "-", "Output file name. - or no designation means STDOUT")
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
			continue
		}

		x := items[0]
		y := items[1]

		simval, err := model.Similarity(x, y)

		fmt.Fprintf(outf, "%s\t%s", x, y)
		if err == nil {
			fmt.Fprintf(outf, "\t%f", simval)
		} else {
			fmt.Fprintf(outf, "\t-")
		}
		fmt.Fprintf(outf, "\n")
	}

}
