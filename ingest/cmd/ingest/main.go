// ingest streams a Wikipedia pages-articles .bz2 dump and writes passages as
// JSON lines, in page order, so two runs over the same dump are byte-identical.
package main

import (
	"bufio"
	"compress/bzip2"
	"encoding/json"
	"encoding/xml"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"

	"github.com/bharathkumardev1/wiki-retrieval/ingest/internal/passages"
	"github.com/bharathkumardev1/wiki-retrieval/ingest/internal/wikitext"
)

type page struct {
	Title    string    `xml:"title"`
	NS       int       `xml:"ns"`
	ID       int64     `xml:"id"`
	Redirect *struct{} `xml:"redirect"`
	Text     string    `xml:"revision>text"`
}

type job struct {
	seq int
	p   page
}

type result struct {
	seq int
	out []passages.Passage
}

type stats struct {
	Pages, Redirects, OtherNS, Empty, Passages int
}

func main() {
	in := flag.String("in", "", "path to .xml.bz2 dump")
	out := flag.String("out", "", "output .jsonl path")
	limit := flag.Int("limit", 0, "stop after this many article pages (0 = all)")
	flag.Parse()
	if *in == "" || *out == "" {
		fmt.Fprintln(os.Stderr, "usage: ingest -in dump.xml.bz2 -out passages.jsonl [-limit N]")
		os.Exit(2)
	}
	if err := run(*in, *out, *limit); err != nil {
		fmt.Fprintln(os.Stderr, "ingest:", err)
		os.Exit(1)
	}
}

func run(inPath, outPath string, limit int) error {
	f, err := os.Open(inPath)
	if err != nil {
		return err
	}
	defer f.Close()
	dec := xml.NewDecoder(bufio.NewReaderSize(bzip2.NewReader(f), 1<<20))

	jobs := make(chan job, 256)
	results := make(chan result, 256)
	var wg sync.WaitGroup
	for i := 0; i < runtime.NumCPU(); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				paras := wikitext.Paragraphs(j.p.Text)
				results <- result{j.seq, passages.Split(j.p.ID, j.p.Title, paras)}
			}
		}()
	}
	go func() { wg.Wait(); close(results) }()

	var st stats
	w, err := os.Create(outPath)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(w, 1<<20)
	collectDone := make(chan error, 1)
	go func() {
		// restore page order: workers finish out of order
		pending := map[int][]passages.Passage{}
		next := 0
		enc := json.NewEncoder(bw)
		enc.SetEscapeHTML(false)
		for r := range results {
			pending[r.seq] = r.out
			for {
				ps, ok := pending[next]
				if !ok {
					break
				}
				delete(pending, next)
				next++
				if len(ps) == 0 {
					st.Empty++
				}
				for _, p := range ps {
					if err := enc.Encode(p); err != nil {
						collectDone <- err
						return
					}
					st.Passages++
				}
			}
		}
		collectDone <- nil
	}()

	seq := 0
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("xml: %w", err)
		}
		se, ok := tok.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}
		var p page
		if err := dec.DecodeElement(&p, &se); err != nil {
			return fmt.Errorf("page decode: %w", err)
		}
		switch {
		case p.NS != 0:
			st.OtherNS++
		case p.Redirect != nil:
			st.Redirects++
		default:
			jobs <- job{seq, p}
			seq++
			st.Pages++
		}
		if limit > 0 && seq >= limit {
			break
		}
	}
	close(jobs)
	if err := <-collectDone; err != nil {
		return err
	}
	if err := bw.Flush(); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	b, _ := json.Marshal(st)
	fmt.Println(string(b))
	return nil
}
