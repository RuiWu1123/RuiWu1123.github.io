# ruiwu1123.github.io

My personal website. React + Vite, deployed to GitHub Pages by an Action on every push to `main`.

## Layout

- `pages/` — Home, Alignment, Blogs, Travel Gallery
- `constants.ts` — publications, news, blog registry, alignment field news
- `public/blogs/` — posts as `<id>.md` and `<id>.zh.md`, figures in `public/blogs/images/`

## Local development

```bash
npm install
npm run dev     # http://localhost:3000
```

Before pushing:

```bash
npx tsc --noEmit -p tsconfig.json
npm run build
```

`tsc` is not optional — `vite build` only transpiles and will not catch undefined identifiers.
