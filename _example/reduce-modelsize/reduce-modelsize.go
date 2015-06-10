package main

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"fmt"
	"github.com/jessevdk/go-flags"
	"io/ioutil"
	"log"
	"os"
)

func getWordList(ifname string) (map[string]struct{}, error) {
	f, err := os.Open(ifname)
	if err != nil {
		return nil, err
	}
	r := bufio.NewReader(f)

	wordlist := map[string]struct{}{}
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		wordlist[scanner.Text()] = struct{}{}
	}
	return wordlist, nil
}

type cmdOptions struct {
	Help   bool   `short:"h" long:"help" description:"Show this help message"`
	Input  string `short:"i" long:"input"`
	Model  string `short:"m" long:"model"`
	Output string `short:"o" long:"output"`
	Log    bool   `long:"log" description:"Enable logging" default:"false"`
}

func main() {
	opts := cmdOptions{}
	optparser := flags.NewParser(&opts, flags.Default)
	optparser.Name = ""
	optparser.Usage = "-i input -o output [OPTIONS]"
	optparser.Parse()

	//show help
	if len(os.Args) == 1 {
		optparser.WriteHelp(os.Stdout)
		os.Exit(0)
	}
	for _, arg := range os.Args {
		if arg == "-h" {
			os.Exit(0)
		}
	}

	if opts.Log == false {
		log.SetOutput(ioutil.Discard)
	}

	log.Printf("Getting wordlist: %s", opts.Input)
	new_words, err := getWordList(opts.Input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}

	log.Printf("Opening model: %s", opts.Model)
	mf, err := os.Open(opts.Model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}
	reader := bufio.NewReader(mf)

	log.Printf("Start creating")
	//start convert
	var vocabSize int
	var vectorSize int
	fmt.Fscanln(reader, &vocabSize, &vectorSize)

	log.Printf("new_words: %d", len(new_words))

	var myword string
	tmp := make([]float32, vectorSize)
	log.Printf("Total: %d", vocabSize)
	tmpout := new(bytes.Buffer)
	count := 0
	for i := 0; i < vocabSize; i++ {
		if i%100000 == 0 {
			log.Printf(" %d finished", i)
		}
		bytes, err := reader.ReadBytes(' ')
		if err != nil {
			fmt.Fprintf(os.Stderr, "%q\n", err)
			os.Exit(1)
		}
		myword = string(bytes[1 : len(bytes)-1])

		binary.Read(reader, binary.LittleEndian, tmp)
		if _, ok := new_words[myword]; ok {
			count++
			tmpout.WriteString(myword)
			tmpout.WriteString(" ")
			binary.Write(tmpout, binary.LittleEndian, tmp)
		}
	}
	log.Printf("done: %d", vocabSize)

	//output
	outf, err := os.Create(opts.Output)
	defer outf.Close()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%q\n", err)
		os.Exit(1)
	}
	w := bufio.NewWriter(outf)
	defer w.Flush()
	b, err := fmt.Fprintln(w, count, vectorSize)
	log.Printf(" Write byte: %d,  err = %v", b, err)
	w.Write(tmpout.Bytes())
	w.Flush()

}
