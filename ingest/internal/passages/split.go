// Package passages packs cleaned paragraphs into retrieval-sized passages.
package passages

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

const (
	TargetWords = 160
	MinWords    = 25
)

type Passage struct {
	ID      string `json:"id"`
	PageID  int64  `json:"page_id"`
	Title   string `json:"title"`
	Index   int    `json:"idx"`
	Text    string `json:"text"`
	Words   int    `json:"words"`
	Content string `json:"sha"`
}

func words(s string) int { return len(strings.Fields(s)) }

// Split greedily packs paragraphs up to TargetWords. A paragraph longer than
// the target is cut on sentence-ish boundaries. IDs depend only on page id and
// position, so re-ingesting an unchanged page yields identical ids, and the
// content hash flags the ones whose text changed.
func Split(pageID int64, title string, paras []string) []Passage {
	var chunks []string
	var cur []string
	n := 0
	flush := func() {
		if n >= MinWords {
			chunks = append(chunks, strings.Join(cur, " "))
		}
		cur, n = nil, 0
	}
	for _, p := range paras {
		for _, piece := range cut(p) {
			w := words(piece)
			if n > 0 && n+w > TargetWords {
				flush()
			}
			cur = append(cur, piece)
			n += w
		}
	}
	flush()
	out := make([]Passage, len(chunks))
	for i, c := range chunks {
		sum := sha256.Sum256([]byte(c))
		out[i] = Passage{
			ID:      fmt.Sprintf("%d-%d", pageID, i),
			PageID:  pageID,
			Title:   title,
			Index:   i,
			Text:    c,
			Words:   words(c),
			Content: hex.EncodeToString(sum[:8]),
		}
	}
	return out
}

func cut(p string) []string {
	if words(p) <= TargetWords {
		return []string{p}
	}
	var out []string
	var cur []string
	n := 0
	for _, w := range strings.Fields(p) {
		cur = append(cur, w)
		n++
		if n >= TargetWords && strings.HasSuffix(w, ".") {
			out = append(out, strings.Join(cur, " "))
			cur, n = nil, 0
		}
	}
	if len(cur) > 0 {
		out = append(out, strings.Join(cur, " "))
	}
	return out
}
