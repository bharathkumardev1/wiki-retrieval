// Package wikitext turns MediaWiki markup into plain paragraphs. It is a
// pragmatic stripper, not a parser: it targets retrieval text quality, and
// anything it cannot place cleanly is dropped rather than guessed at.
package wikitext

import (
	"html"
	"regexp"
	"strings"
)

var (
	comment    = regexp.MustCompile(`(?s)<!--.*?-->`)
	refPair    = regexp.MustCompile(`(?s)<ref[^>/]*>.*?</ref>`)
	refSelf    = regexp.MustCompile(`<ref[^>]*/>`)
	tag        = regexp.MustCompile(`</?[a-zA-Z][^>]*>`)
	fileLink   = regexp.MustCompile(`(?i)^\[\[(file|image|category):`)
	pipedLink  = regexp.MustCompile(`\[\[(?:[^\]|]*\|)?([^\]]*)\]\]`)
	extLink    = regexp.MustCompile(`\[https?://[^\s\]]+\s?([^\]]*)\]`)
	quotes     = regexp.MustCompile(`'{2,5}`)
	heading    = regexp.MustCompile(`^=+\s*(.*?)\s*=+\s*$`)
	multiSpace = regexp.MustCompile(`[ \t]+`)
)

// stripNested removes balanced open...close spans, handling nesting. Used for
// templates {{...}} and tables {|...|}.
func stripNested(s, open, closing string) string {
	var b strings.Builder
	depth := 0
	for i := 0; i < len(s); {
		switch {
		case strings.HasPrefix(s[i:], open):
			depth++
			i += len(open)
		case depth > 0 && strings.HasPrefix(s[i:], closing):
			depth--
			i += len(closing)
		case depth == 0:
			b.WriteByte(s[i])
			i++
		default:
			i++
		}
	}
	return b.String()
}

// Paragraphs returns the cleaned body text split on blank lines. Section
// headings are dropped from the text; they are not passages.
func Paragraphs(raw string) []string {
	s := comment.ReplaceAllString(raw, "")
	s = refPair.ReplaceAllString(s, "")
	s = refSelf.ReplaceAllString(s, "")
	s = stripNested(s, "{{", "}}")
	s = stripNested(s, "{|", "|}")
	var out []string
	for _, block := range strings.Split(s, "\n\n") {
		var keep []string
		for _, line := range strings.Split(block, "\n") {
			line = strings.TrimSpace(line)
			if line == "" || heading.MatchString(line) || fileLink.MatchString(line) {
				continue
			}
			// list items and indents are navigation or fragments, not prose
			if strings.HasPrefix(line, "*") || strings.HasPrefix(line, "#") ||
				strings.HasPrefix(line, ":") || strings.HasPrefix(line, ";") {
				continue
			}
			keep = append(keep, line)
		}
		p := strings.Join(keep, " ")
		p = pipedLink.ReplaceAllString(p, "$1")
		p = extLink.ReplaceAllString(p, "$1")
		p = tag.ReplaceAllString(p, "")
		p = strings.ReplaceAll(html.UnescapeString(p), "\u00a0", " ")
		p = quotes.ReplaceAllString(p, "")
		p = multiSpace.ReplaceAllString(p, " ")
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
