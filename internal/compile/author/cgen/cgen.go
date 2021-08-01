package cgen

import "strconv"

const (
	aligned             = "aligned"
	ampersand           = "&"
	arrow               = "->"
	assign              = "="
	asterisk            = "*"
	attribute           = "__attribute__"
	bang                = "!"
	brace1              = "{"
	brace2              = "}"
	break_              = "break"
	calloc              = "calloc"
	caret               = "^"
	case_               = "case"
	char                = "char"
	cmpE                = "=="
	cmpG                = ">"
	cmpGE               = ">="
	cmpL                = "<"
	cmpLE               = "<="
	cmpNE               = "!="
	colon               = ":"
	comma               = ","
	continue_           = "continue"
	cplusplus           = "__cplusplus"
	cpuSupports         = "__builtin_cpu_supports"
	ctzl                = "__builtin_ctzl"
	dec                 = "--"
	default_            = "default"
	dot                 = "."
	doubleQuote         = "\""
	ellipsis            = "..."
	else_               = "else"
	empty               = ""
	endif               = "endif"
	errno               = "errno"
	expect              = "__builtin_expect"
	extern              = "extern"
	float               = "float"
	floatSuffix         = "f"
	for_                = "for"
	free                = "free"
	gap                 = "/**/"
	goto_               = "goto"
	hash                = "#"
	ifdef               = "ifdef"
	if_                 = "if"
	inc                 = "++"
	include             = "include"
	int64T              = "int64_t"
	int_                = "int"
	land                = "&&"
	lineNum             = "__LINE__"
	linkageC            = "C"
	long                = "long"
	lor                 = "||"
	malloc              = "malloc"
	memcpy              = "memcpy"
	memset              = "memset"
	minus               = "-"
	newline             = "\n"
	once                = "once"
	one                 = "1"
	packed              = "packed"
	paren1              = "("
	paren2              = ")"
	percent             = "%"
	pipe                = "|"
	plus                = "+"
	pragma              = "pragma"
	pthreadCondDestroy  = "pthread_cond_destroy"
	pthreadCondInit     = "pthread_cond_init"
	pthreadCondSignal   = "pthread_cond_signal"
	pthreadCondT        = "pthread_cond_t"
	pthreadCondWait     = "pthread_cond_wait"
	pthreadCreate       = "pthread_create"
	pthreadJoin         = "pthread_join"
	pthreadMutexDestroy = "pthread_mutex_destroy"
	pthreadMutexInit    = "pthread_mutex_init"
	pthreadMutexLock    = "pthread_mutex_lock"
	pthreadMutexT       = "pthread_mutex_t"
	pthreadMutexUnlock  = "pthread_mutex_unlock"
	pthreadT            = "pthread_t"
	ptrdiffT            = "ptrdiff_t"
	questionMark        = "?"
	restrict            = "restrict"
	return_             = "return"
	semicolon           = ";"
	shiftHigh           = "<<"
	shiftLow            = ">>"
	sizeof              = "sizeof"
	sizeT               = "size_t"
	slash               = "/"
	slashes             = "//"
	space               = " "
	sprintf             = "sprintf"
	squareBracket1      = "["
	squareBracket2      = "]"
	static              = "static"
	struct_             = "struct"
	switch_             = "switch"
	tilde               = "~"
	typedef             = "typedef"
	vaEnd               = "va_end"
	vaList              = "va_list"
	vaStart             = "va_start"
	void                = "void"
	vsnprintf           = "vsnprintf"
	zero                = "0"
)

type Add struct {
	Expr1, Expr2 Gen
}

func (a Add) Append(to []byte) []byte {
	to = a.Expr1.Append(to)
	to = append(to, plus...)
	to = a.Expr2.Append(to)
	return to
}

type AddAssign struct {
	Expr1, Expr2 Gen
}

func (a AddAssign) Append(to []byte) []byte {
	to = a.Expr1.Append(to)
	to = append(to, space+plus+assign+space...)
	to = a.Expr2.Append(to)
	return to
}

type Addr struct {
	Expr Gen
}

func (a Addr) Append(to []byte) []byte {
	to = append(to, ampersand...)
	to = a.Expr.Append(to)
	return to
}

type AddrArrow Arrow

func (a AddrArrow) Append(to []byte) []byte {
	to = Addr{Arrow(a)}.Append(to)
	return to
}

type AddrDot Dot

func (a AddrDot) Append(to []byte) []byte {
	to = Addr{Dot(a)}.Append(to)
	return to
}

type Aligned int

func (a Aligned) Append(to []byte) []byte {
	to = append(to, aligned...)
	to = Paren{IntLit(a)}.Append(to)
	return to
}

type And struct {
	Expr1, Expr2 Gen
}

func (a And) Append(to []byte) []byte {
	to = a.Expr1.Append(to)
	to = append(to, ampersand...)
	to = a.Expr2.Append(to)
	return to
}

type AndAssign struct {
	Expr1, Expr2 Gen
}

func (a AndAssign) Append(to []byte) []byte {
	to = a.Expr1.Append(to)
	to = append(to, space+ampersand+assign+space...)
	to = a.Expr2.Append(to)
	return to
}

type AngleBracketed string

func (a AngleBracketed) Append(to []byte) []byte {
	to = append(to, cmpL...)
	to = append(to, a...)
	to = append(to, cmpG...)
	return to
}

type Arrow struct {
	Expr Gen
	Name string
}

func (a Arrow) Append(to []byte) []byte {
	to = a.Expr.Append(to)
	to = append(to, arrow...)
	to = append(to, a.Name...)
	return to
}

type Assign struct {
	Expr1, Expr2 Gen
}

func (a Assign) Append(to []byte) []byte {
	to = a.Expr1.Append(to)
	to = append(to, space+assign+space...)
	to = a.Expr2.Append(to)
	return to
}

type At struct {
	Expr Gen
}

func (a At) Append(to []byte) []byte {
	to = append(to, asterisk...)
	to = a.Expr.Append(to)
	return to
}

type AttrSpec struct {
	Attrs Gen
}

func (a AttrSpec) Append(to []byte) []byte {
	to = append(to, attribute...)
	to = Paren{Paren{a.Attrs}}.Append(to)
	return to
}

type Block struct {
	Inner Gen
}

func (b Block) Append(to []byte) []byte {
	to = append(to, brace1+newline...)
	to = Maybe{b.Inner}.Append(to)
	to = append(to, brace2...)
	return to
}

type Brace struct {
	Inner Gen
}

func (b Brace) Append(to []byte) []byte {
	to = append(to, brace1...)
	to = Maybe{b.Inner}.Append(to)
	to = append(to, brace2...)
	return to
}

type Call struct {
	Func, Args Gen
}

func (c Call) Append(to []byte) []byte {
	to = c.Func.Append(to)
	to = Paren{c.Args}.Append(to)
	return to
}

type Case struct {
	Expr, Body Gen
}

func (c Case) Append(to []byte) []byte {
	if c.Expr == nil {
		to = append(to, default_...)
	} else {
		to = append(to, case_+space...)
		to = c.Expr.Append(to)
	}
	to = append(to, colon...)
	if c.Body != nil {
		to = append(to, space...)
		to = Block{c.Body}.Append(to)
	}
	return to
}

type Cast struct {
	Type, Expr Gen
}

func (c Cast) Append(to []byte) []byte {
	to = Paren{c.Type}.Append(to)
	to = c.Expr.Append(to)
	return to
}

type CmpE struct {
	Expr1, Expr2 Gen
}

func (c CmpE) Append(to []byte) []byte {
	to = c.Expr1.Append(to)
	to = append(to, space+cmpE+space...)
	to = c.Expr2.Append(to)
	return to
}

type CmpG struct {
	Expr1, Expr2 Gen
}

func (c CmpG) Append(to []byte) []byte {
	to = c.Expr1.Append(to)
	to = append(to, space+cmpG+space...)
	to = c.Expr2.Append(to)
	return to
}

type CmpGE struct {
	Expr1, Expr2 Gen
}

func (c CmpGE) Append(to []byte) []byte {
	to = c.Expr1.Append(to)
	to = append(to, space+cmpGE+space...)
	to = c.Expr2.Append(to)
	return to
}

type CmpL struct {
	Expr1, Expr2 Gen
}

func (c CmpL) Append(to []byte) []byte {
	to = c.Expr1.Append(to)
	to = append(to, space+cmpL+space...)
	to = c.Expr2.Append(to)
	return to
}

type CmpLE struct {
	Expr1, Expr2 Gen
}

func (c CmpLE) Append(to []byte) []byte {
	to = c.Expr1.Append(to)
	to = append(to, space+cmpLE+space...)
	to = c.Expr2.Append(to)
	return to
}

type CmpNE struct {
	Expr1, Expr2 Gen
}

func (c CmpNE) Append(to []byte) []byte {
	to = c.Expr1.Append(to)
	to = append(to, space+cmpNE+space...)
	to = c.Expr2.Append(to)
	return to
}

type CommaLines []Gen

func (c CommaLines) Append(to []byte) []byte {
	first := true
	for _, gen := range c {
		if gen == nil {
			continue
		}
		if first {
			first = false
		} else {
			to = append(to, comma...)
		}
		to = append(to, newline...)
		to = gen.Append(to)
	}
	if !first {
		to = append(to, newline...)
	}
	return to
}

type CommaSpaced []Gen

func (c CommaSpaced) Append(to []byte) []byte {
	first := true
	for _, gen := range c {
		if gen == nil {
			continue
		}
		if first {
			first = false
		} else {
			to = append(to, comma+space...)
		}
		to = gen.Append(to)
	}
	return to
}

type Comment []string

func (c Comment) Append(to []byte) []byte {
	for _, line := range c {
		switch line {
		case empty:
			to = append(to, slashes+newline...)
		default:
			to = append(to, slashes+space...)
			to = append(to, line...)
			to = append(to, newline...)
		}
	}
	return to
}

type DecPost struct {
	Expr Gen
}

func (d DecPost) Append(to []byte) []byte {
	to = d.Expr.Append(to)
	to = append(to, dec...)
	return to
}

type DecPre struct {
	Expr Gen
}

func (d DecPre) Append(to []byte) []byte {
	to = append(to, dec...)
	to = d.Expr.Append(to)
	return to
}

type Directive string

const (
	Endif   Directive = endif
	Ifdef   Directive = ifdef
	Include Directive = include
	Pragma  Directive = pragma
)

type Dot struct {
	Expr Gen
	Name string
}

func (d Dot) Append(to []byte) []byte {
	to = d.Expr.Append(to)
	to = append(to, dot...)
	to = append(to, d.Name...)
	return to
}

type DoubleQuoted string

func (d DoubleQuoted) Append(to []byte) []byte {
	to = append(to, doubleQuote...)
	to = append(to, d...)
	to = append(to, doubleQuote...)
	return to
}

type Elem struct {
	Arr, Idx Gen
}

func (e Elem) Append(to []byte) []byte {
	to = e.Arr.Append(to)
	to = append(to, squareBracket1...)
	to = Maybe{e.Idx}.Append(to)
	to = append(to, squareBracket2...)
	return to
}

type Extern struct {
	Tail Gen
}

func (e Extern) Append(to []byte) []byte {
	to = append(to, extern+space...)
	to = e.Tail.Append(to)
	return to
}

type Field struct {
	Type, What Gen
}

func (f Field) Append(to []byte) []byte {
	to = f.Type.Append(to)
	to = append(to, space...)
	to = f.What.Append(to)
	to = append(to, semicolon...)
	return to
}

type FloatLit float64

func (f FloatLit) Append(to []byte) []byte {
	to = strconv.AppendFloat(to, float64(f), 'e', -1, 32)
	to = append(to, floatSuffix...)
	return to
}

type For struct {
	Init, Cond, Post, Body Gen
}

func (f For) Append(to []byte) []byte {
	to = append(to, for_+space+paren1...)
	to = Maybe{f.Init}.Append(to)
	if to[len(to)-1] != semicolon[0] {
		to = append(to, semicolon...)
	}
	to = append(to, space...)
	to = Maybe{f.Cond}.Append(to)
	to = append(to, semicolon+space...)
	to = Maybe{f.Post}.Append(to)
	to = append(to, paren2...)
	if f.Body != nil {
		to = append(to, space...)
		to = Block{f.Body}.Append(to)
	}
	return to
}

type FuncDecl struct {
	ReturnType Gen
	Name       string
	Params     Gen
}

func (f FuncDecl) Append(to []byte) []byte {
	to = f.ReturnType.Append(to)
	to = append(to, space...)
	to = Call{Vb(f.Name), f.Params}.Append(to)
	to = append(to, semicolon+newline...)
	return to
}

type FuncDef struct {
	ReturnType Gen
	Name       string
	Params     Gen
	Body       Gen
}

func (f FuncDef) Append(to []byte) []byte {
	var g1, g2, g3 Gen
	g1 = f.ReturnType
	g2 = Call{Vb(f.Name), f.Params}
	g3 = Block{f.Body}
	to = Spaced{g1, g2, g3}.Append(to)
	to = append(to, newline...)
	return to
}

type Gen interface {
	Append(to []byte) []byte
}

type Gens []Gen

func (gs Gens) Append(to []byte) []byte {
	for _, gen := range gs {
		if gen != nil {
			to = gen.Append(to)
		}
	}
	return to
}

type Goto Label

func (g Goto) Append(to []byte) []byte {
	to = append(to, goto_+space...)
	to = append(to, g...)
	return to
}

type If struct {
	Cond Gen
	Then Stmts
	Else Stmts
}

func (i If) Append(to []byte) []byte {
	to = append(to, if_+space...)
	to = Paren{i.Cond}.Append(to)
	to = append(to, space...)
	to = Block{i.Then}.Append(to)
	if n := len(i.Else); n != 0 {
		to = append(to, space+else_+space...)
		chain := false
		if n == 1 {
			_, chain = i.Else[0].(If)
		}
		if chain {
			to = i.Else[0].Append(to)
		} else {
			to = Block{i.Else}.Append(to)
		}
	}
	return to
}

type If1 struct {
	Cond, Then, Else Gen
}

func (i If1) Append(to []byte) []byte {
	to = append(to, if_+space...)
	to = Paren{i.Cond}.Append(to)
	to = append(to, space...)
	if i.Else == nil {
		to = i.Then.Append(to)
	} else {
		to = Stmts{i.Then}.Append(to)
		to = append(to, else_+space...)
		to = i.Else.Append(to)
	}
	return to
}

type IncPost struct {
	Expr Gen
}

func (i IncPost) Append(to []byte) []byte {
	to = i.Expr.Append(to)
	to = append(to, inc...)
	return to
}

type IncPre struct {
	Expr Gen
}

func (i IncPre) Append(to []byte) []byte {
	to = append(to, inc...)
	to = i.Expr.Append(to)
	return to
}

type IntLit int

func (i IntLit) Append(to []byte) []byte {
	to = strconv.AppendInt(to, int64(i), 10)
	return to
}

type IsNonzero struct {
	Expr Gen
}

func (i IsNonzero) Append(to []byte) []byte {
	to = append(to, bang+bang...)
	to = i.Expr.Append(to)
	return to
}

type IsZero struct {
	Expr Gen
}

func (i IsZero) Append(to []byte) []byte {
	to = append(to, bang...)
	to = i.Expr.Append(to)
	return to
}

type Label string

func (l Label) Append(to []byte) []byte {
	to = append(to, l...)
	to = append(to, colon...)
	return to
}

type Land struct {
	Expr1, Expr2 Gen
}

func (l Land) Append(to []byte) []byte {
	to = l.Expr1.Append(to)
	to = append(to, space+land+space...)
	to = l.Expr2.Append(to)
	return to
}

type Lor struct {
	Expr1, Expr2 Gen
}

func (l Lor) Append(to []byte) []byte {
	to = l.Expr1.Append(to)
	to = append(to, space+lor+space...)
	to = l.Expr2.Append(to)
	return to
}

type Maybe struct {
	What Gen
}

func (m Maybe) Append(to []byte) []byte {
	if m.What != nil {
		to = m.What.Append(to)
	}
	return to
}

type MaybeSpace struct {
	What Gen
}

func (m MaybeSpace) Append(to []byte) []byte {
	if m.What != nil {
		to = append(to, space...)
		to = m.What.Append(to)
	}
	return to
}

type Mul struct {
	Expr1, Expr2 Gen
}

func (m Mul) Append(to []byte) []byte {
	to = m.Expr1.Append(to)
	to = append(to, asterisk...)
	to = m.Expr2.Append(to)
	return to
}

type MulAssign struct {
	Expr1, Expr2 Gen
}

func (m MulAssign) Append(to []byte) []byte {
	to = m.Expr1.Append(to)
	to = append(to, space+asterisk+assign+space...)
	to = m.Expr2.Append(to)
	return to
}

type Neg struct {
	Expr Gen
}

func (n Neg) Append(to []byte) []byte {
	to = append(to, minus...)
	to = n.Expr.Append(to)
	return to
}

type Not struct {
	Expr Gen
}

func (n Not) Append(to []byte) []byte {
	to = append(to, tilde...)
	to = n.Expr.Append(to)
	return to
}

type Or struct {
	Expr1, Expr2 Gen
}

func (o Or) Append(to []byte) []byte {
	to = o.Expr1.Append(to)
	to = append(to, pipe...)
	to = o.Expr2.Append(to)
	return to
}

type OrAssign struct {
	Expr1, Expr2 Gen
}

func (o OrAssign) Append(to []byte) []byte {
	to = o.Expr1.Append(to)
	to = append(to, space+pipe+assign+space...)
	to = o.Expr2.Append(to)
	return to
}

type Param struct {
	Type, What Gen
}

func (p Param) Append(to []byte) []byte {
	to = p.Type.Append(to)
	to = append(to, space...)
	to = p.What.Append(to)
	return to
}

type Paren struct {
	Inner Gen
}

func (p Paren) Append(to []byte) []byte {
	to = append(to, paren1...)
	to = Maybe{p.Inner}.Append(to)
	to = append(to, paren2...)
	return to
}

type Preprocessor struct {
	Head Directive
	Tail Gen
}

func (p Preprocessor) Append(to []byte) []byte {
	to = append(to, hash...)
	to = append(to, p.Head...)
	to = MaybeSpace{p.Tail}.Append(to)
	to = append(to, newline...)
	return to
}

type Ptr struct {
	Type Gen
}

func (p Ptr) Append(to []byte) []byte {
	to = p.Type.Append(to)
	to = append(to, asterisk...)
	return to
}

type Quo struct {
	Expr1, Expr2 Gen
}

func (q Quo) Append(to []byte) []byte {
	to = q.Expr1.Append(to)
	to = append(to, slash...)
	to = q.Expr2.Append(to)
	return to
}

type QuoAssign struct {
	Expr1, Expr2 Gen
}

func (q QuoAssign) Append(to []byte) []byte {
	to = q.Expr1.Append(to)
	to = append(to, space+slash+assign+space...)
	to = q.Expr2.Append(to)
	return to
}

type Rem struct {
	Expr1, Expr2 Gen
}

func (r Rem) Append(to []byte) []byte {
	to = r.Expr1.Append(to)
	to = append(to, percent...)
	to = r.Expr2.Append(to)
	return to
}

type RemAssign struct {
	Expr1, Expr2 Gen
}

func (r RemAssign) Append(to []byte) []byte {
	to = r.Expr1.Append(to)
	to = append(to, space+percent+assign+space...)
	to = r.Expr2.Append(to)
	return to
}

type RestrictPtr Ptr

func (r RestrictPtr) Append(to []byte) []byte {
	to = Ptr(r).Append(to)
	to = append(to, restrict...)
	return to
}

type Return struct {
	Expr Gen
}

func (r Return) Append(to []byte) []byte {
	to = append(to, return_...)
	to = MaybeSpace{r.Expr}.Append(to)
	return to
}

type ShiftHigh struct {
	Expr1, Expr2 Gen
}

func (s ShiftHigh) Append(to []byte) []byte {
	to = s.Expr1.Append(to)
	to = append(to, shiftHigh...)
	to = s.Expr2.Append(to)
	return to
}

type ShiftHighAssign struct {
	Expr1, Expr2 Gen
}

func (s ShiftHighAssign) Append(to []byte) []byte {
	to = s.Expr1.Append(to)
	to = append(to, space+shiftHigh+assign+space...)
	to = s.Expr2.Append(to)
	return to
}

type ShiftLow struct {
	Expr1, Expr2 Gen
}

func (s ShiftLow) Append(to []byte) []byte {
	to = s.Expr1.Append(to)
	to = append(to, shiftLow...)
	to = s.Expr2.Append(to)
	return to
}

type ShiftLowAssign struct {
	Expr1, Expr2 Gen
}

func (s ShiftLowAssign) Append(to []byte) []byte {
	to = s.Expr1.Append(to)
	to = append(to, space+shiftLow+assign+space...)
	to = s.Expr2.Append(to)
	return to
}

type Sizeof struct {
	What Gen
}

func (s Sizeof) Append(to []byte) []byte {
	to = append(to, sizeof...)
	to = Paren{s.What}.Append(to)
	return to
}

type Spaced []Gen

func (s Spaced) Append(to []byte) []byte {
	first := true
	for _, gen := range s {
		if gen == nil {
			continue
		}
		if first {
			first = false
		} else {
			to = append(to, space...)
		}
		to = gen.Append(to)
	}
	return to
}

type Static struct {
	Tail Gen
}

func (s Static) Append(to []byte) []byte {
	to = append(to, static+space...)
	to = s.Tail.Append(to)
	return to
}

type StaticFuncDef FuncDef

func (s StaticFuncDef) Append(to []byte) []byte {
	to = Static{FuncDef(s)}.Append(to)
	return to
}

type Stmts []Gen

func (s Stmts) Append(to []byte) []byte {
	for _, gen := range s {
		if gen == nil {
			continue
		}
		n1 := len(to)
		to = gen.Append(to)
		n2 := len(to)
		if n1 >= n2 {
			continue
		}
		switch to[n2-1] {
		case newline[0]:
		case brace2[0], semicolon[0]:
			to = append(to, newline...)
		default:
			to = append(to, semicolon+newline...)
		}
	}
	return to
}

type StructDef struct {
	Name   string
	Fields Gen
	Attrs  Gen
}

func (s StructDef) Append(to []byte) []byte {
	var g1, g2, g3 Gen
	g1, g2 = StructTag(s.Name), Block{s.Fields}
	if s.Attrs != nil {
		g3 = AttrSpec{s.Attrs}
	}
	to = Spaced{g1, g2, g3}.Append(to)
	to = append(to, semicolon+newline...)
	return to
}

type StructFwd string

func (s StructFwd) Append(to []byte) []byte {
	var g1, g2 Gen
	g1, g2 = StructTag(s), Vb(s)
	to = Typedef{g1, g2}.Append(to)
	return to
}

type StructTag string

func (s StructTag) Append(to []byte) []byte {
	to = append(to, struct_+space...)
	to = append(to, s...)
	return to
}

type Sub struct {
	Expr1, Expr2 Gen
}

func (s Sub) Append(to []byte) []byte {
	to = s.Expr1.Append(to)
	to = append(to, minus...)
	to = s.Expr2.Append(to)
	return to
}

type SubAssign struct {
	Expr1, Expr2 Gen
}

func (s SubAssign) Append(to []byte) []byte {
	to = s.Expr1.Append(to)
	to = append(to, space+minus+assign+space...)
	to = s.Expr2.Append(to)
	return to
}

type Switch struct {
	Expr, Cases Gen
}

func (s Switch) Append(to []byte) []byte {
	to = append(to, switch_+space...)
	to = Paren{s.Expr}.Append(to)
	to = append(to, space...)
	to = Block{s.Cases}.Append(to)
	return to
}

type Table struct {
	Flat []Gen
	Cols int
}

func (t Table) Append(to []byte) []byte {
	last := t.Cols - 1
	if last < 0 {
		return to
	}
	var text []byte
	sizes := make([]int, 0, len(t.Flat))
	maxes := make([]int, last)
	col := 0
	for i := range t.Flat {
		if col == last {
			col = 0
			continue
		}
		was := len(text)
		if gen := t.Flat[i]; gen != nil {
			text = gen.Append(text)
		}
		size := len(text) - was
		sizes = append(sizes, size)
		if maxes[col] < size {
			maxes[col] = size
		}
		col += 1
	}
	most := 0
	for _, max := range maxes {
		if most < max {
			most = max
		}
	}
	sp, nl := space[0], newline[0]
	spaces := make([]byte, most+1)
	for i := range spaces {
		spaces[i] = sp
	}
	for i := range t.Flat {
		if col == last {
			was := len(to)
			if gen := t.Flat[i]; gen != nil {
				to = gen.Append(to)
			}
			now := len(to)
			if was >= now || to[now-1] != nl {
				to = append(to, nl)
			}
			col = 0
			continue
		}
		size := sizes[0]
		sizes = sizes[1:]
		to = append(to, text[:size]...)
		text = text[size:]
		fill := maxes[col] - size + 1
		to = append(to, spaces[:fill]...)
		col += 1
	}
	return to
}

type Ternary struct {
	Cond, Then, Else Gen
}

func (t Ternary) Append(to []byte) []byte {
	to = t.Cond.Append(to)
	to = append(to, space+questionMark+space...)
	to = t.Then.Append(to)
	to = append(to, space+colon+space...)
	to = t.Else.Append(to)
	return to
}

type Typedef struct {
	Type, What Gen
}

func (t Typedef) Append(to []byte) []byte {
	to = append(to, typedef+space...)
	to = Var{t.Type, t.What, nil}.Append(to)
	to = append(to, newline...)
	return to
}

type TypedefPtrFunc struct {
	ReturnType, What, Params Gen
}

func (t TypedefPtrFunc) Append(to []byte) []byte {
	var call Gen
	call = Call{Paren{At{t.What}}, t.Params}
	to = Typedef{t.ReturnType, call}.Append(to)
	return to
}

type Unlikely struct {
	Cond Gen
}

func (u Unlikely) Append(to []byte) []byte {
	var args Gen
	args = CommaSpaced{u.Cond, Zero}
	to = Call{Vb(expect), args}.Append(to)
	return to
}

type Var struct {
	Type, What, Init Gen
}

func (v Var) Append(to []byte) []byte {
	to = v.Type.Append(to)
	to = append(to, space...)
	to = v.What.Append(to)
	if v.Init != nil {
		to = append(to, space+assign+space...)
		to = v.Init.Append(to)
	}
	to = append(to, semicolon...)
	return to
}

type Vb string

func (v Vb) Append(to []byte) []byte {
	to = append(to, v...)
	return to
}

type Xor struct {
	Expr1, Expr2 Gen
}

func (x Xor) Append(to []byte) []byte {
	to = x.Expr1.Append(to)
	to = append(to, caret...)
	to = x.Expr2.Append(to)
	return to
}

type XorAssign struct {
	Expr1, Expr2 Gen
}

func (x XorAssign) Append(to []byte) []byte {
	to = x.Expr1.Append(to)
	to = append(to, space+caret+assign+space...)
	to = x.Expr2.Append(to)
	return to
}

var (
	BitsPerByte         Gen = IntLit(8)
	BitsPerLong         Gen = Paren{Mul{Sizeof{Long}, BitsPerByte}}
	Brace1              Gen = Vb(brace1)
	Brace2              Gen = Vb(brace2)
	Break               Gen = Vb(break_)
	Calloc              Gen = Vb(calloc)
	Char                Gen = Vb(char)
	Continue            Gen = Vb(continue_)
	Cplusplus           Gen = Vb(cplusplus)
	CpuSupports         Gen = Vb(cpuSupports)
	Ctzl                Gen = Vb(ctzl)
	Ellipsis            Gen = Vb(ellipsis)
	Errno               Gen = Vb(errno)
	Float               Gen = Vb(float)
	Free                Gen = Vb(free)
	Gap                 Gen = Vb(gap)
	Int64T              Gen = Vb(int64T)
	Int                 Gen = Vb(int_)
	LineNum             Gen = Vb(lineNum)
	LinkageC            Gen = DoubleQuoted(linkageC)
	Long                Gen = Vb(long)
	Malloc              Gen = Vb(malloc)
	Memcpy              Gen = Vb(memcpy)
	Memset              Gen = Vb(memset)
	NegOne              Gen = Neg{One}
	Newline             Gen = Vb(newline)
	Once                Gen = Vb(once)
	One                 Gen = Vb(one)
	Packed              Gen = Vb(packed)
	PragmaOnce          Gen = Preprocessor{Pragma, Once}
	PthreadCondDestroy  Gen = Vb(pthreadCondDestroy)
	PthreadCondInit     Gen = Vb(pthreadCondInit)
	PthreadCondSignal   Gen = Vb(pthreadCondSignal)
	PthreadCondT        Gen = Vb(pthreadCondT)
	PthreadCondWait     Gen = Vb(pthreadCondWait)
	PthreadCreate       Gen = Vb(pthreadCreate)
	PthreadJoin         Gen = Vb(pthreadJoin)
	PthreadMutexDestroy Gen = Vb(pthreadMutexDestroy)
	PthreadMutexInit    Gen = Vb(pthreadMutexInit)
	PthreadMutexLock    Gen = Vb(pthreadMutexLock)
	PthreadMutexT       Gen = Vb(pthreadMutexT)
	PthreadMutexUnlock  Gen = Vb(pthreadMutexUnlock)
	PthreadT            Gen = Vb(pthreadT)
	PtrChar             Gen = Ptr{Char}
	PtrdiffT            Gen = Vb(ptrdiffT)
	PtrFloat            Gen = Ptr{Float}
	PtrInt64T           Gen = Ptr{Int64T}
	PtrPthreadT         Gen = Ptr{PthreadT}
	PtrPtrChar          Gen = Ptr{PtrChar}
	PtrPtrVoid          Gen = Ptr{PtrVoid}
	PtrVoid             Gen = Ptr{Void}
	RestrictPtrChar     Gen = RestrictPtr{Char}
	RestrictPtrFloat    Gen = RestrictPtr{Float}
	RestrictPtrInt64T   Gen = RestrictPtr{Int64T}
	SizeT               Gen = Vb(sizeT)
	Sprintf             Gen = Vb(sprintf)
	VaEnd               Gen = Vb(vaEnd)
	VaList              Gen = Vb(vaList)
	VaStart             Gen = Vb(vaStart)
	Void                Gen = Vb(void)
	Vsnprintf           Gen = Vb(vsnprintf)
	Zero                Gen = Vb(zero)
	Zeros               Gen = Brace{Zero}
)

var Linkage1 Gen = Gens{
	Preprocessor{Ifdef, Cplusplus},
	Extern{Spaced{LinkageC, Brace1, Gap}}, Newline,
	Preprocessor{Endif, nil},
}

var Linkage2 Gen = Gens{
	Preprocessor{Ifdef, Cplusplus},
	Spaced{Gap, Brace2}, Newline,
	Preprocessor{Endif, nil},
}
