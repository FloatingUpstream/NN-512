package email

func Prep() string {
	const (
		begin = "<h2>Email</h2><pre>"
		addr  = "37ef.ced3@gmail.com"
		end   = "</pre>"
	)
	return begin + addr + end
}
