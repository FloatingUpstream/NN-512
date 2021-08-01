package cov

func Rect(area, max1, n1, n2 int) (s1, s2 int) {
	best := -1
	eval := func(t1, t2 int) {
		if t1 > max1 {
			return
		}
		var (
			fit1 = n1 / t1
			rem1 = n1 % t1
			fit2 = n2 / t2
			rem2 = n2 % t2
		)
		cost := 0
		if rem1 > 0 {
			cost += (area - rem1*t2) * fit2
			if rem2 > 0 {
				cost += area - rem1*rem2
			}
		}
		if rem2 > 0 {
			cost += (area - t1*rem2) * fit1
		}
		if best == -1 ||
			best > cost ||
			best == cost && s1 > t1 {
			best = cost
			s1, s2 = t1, t2
		}
	}
	for lo := 1; lo*lo <= area; lo++ {
		hi := area / lo
		if lo*hi == area {
			eval(lo, hi)
			if lo != hi {
				eval(hi, lo)
			}
		}
	}
	return
}

func Box(vol, max1, n1, n2, n3 int) (s1, s2, s3 int) {
	best := -1
	eval := func(t1, t2, t3 int) {
		if t1 > max1 {
			return
		}
		var (
			dim1 = (n1 + t1 - 1) / t1 * t1
			dim2 = (n2 + t2 - 1) / t2 * t2
			dim3 = (n3 + t3 - 1) / t3 * t3
		)
		cost := dim1 * dim2 * dim3
		if best == -1 ||
			best > cost ||
			best == cost && (s1 > t1 ||
				s1 == t1 && s2 > t2) {
			best = cost
			s1, s2, s3 = t1, t2, t3
		}
	}
	for lo := 1; lo*lo*lo <= vol; lo++ {
		area := vol / lo
		if lo*area != vol {
			continue
		}
		for md := lo; md*md <= area; md++ {
			hi := area / md
			if md*hi != area {
				continue
			}
			eval(lo, md, hi)
			if lo != md {
				eval(md, lo, hi)
				eval(md, hi, lo)
			}
			if md != hi {
				eval(lo, hi, md)
				eval(hi, lo, md)
				if lo != md {
					eval(hi, md, lo)
				}
			}
		}
	}
	return
}
