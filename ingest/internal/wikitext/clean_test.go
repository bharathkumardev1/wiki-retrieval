package wikitext

import (
	"reflect"
	"testing"
)

func TestParagraphs(t *testing.T) {
	raw := "{{Infobox|a={{nested}}|b=2}}\n'''Paris''' is the [[capital city|capital]] of [[France]].<ref>cite</ref>\n\n== History ==\n\n{| class=\"wikitable\"\n|a||b\n|}\n\nIt has [http://x.org a site] <b>online</b>.\n[[File:Pic.jpg|thumb|caption]]\n* list item\n"
	got := Paragraphs(raw)
	want := []string{"Paris is the capital of France.", "It has a site online."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestUnbalancedTemplateDoesNotPanic(t *testing.T) {
	_ = Paragraphs("text {{never closed and more text")
}

func TestEntitiesAreDecoded(t *testing.T) {
	got := Paragraphs("It is 120&nbsp;m &amp; more.")
	if len(got) != 1 || got[0] != "It is 120 m & more." {
		t.Fatalf("got %q", got)
	}
}
