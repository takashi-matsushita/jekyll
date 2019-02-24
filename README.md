git clone ssh://git@github.com/takashi-matsushita/jekyll
cd jekyll
git checkout -b gh-pages
cd ..
jekyll new jekyll --force
cd jekyll
bundle update
jekyll build
jekyll serve
git add .
git commit -m "initial import of jekyll"
git push origin gh-pages
git tag -a minima-initial -m "initial setting working on github"
git push origin minima-initial

bundle exec jekyll serve
