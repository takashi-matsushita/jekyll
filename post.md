---
layout: default
---

{% for post in site.posts %}

  {% assign currentdate = post.date | date: "%B %Y" %}
  {% if currentdate != date %}
  <h4 id="y{{post.date | date: "%Y"}}"> {{ currentdate }} </h4>
    {% assign date = currentdate %}
  {% endif %}

&bull; {{ post.date | date: "%d" }}&nbsp; - &nbsp;<a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a> &nbsp; - &nbsp; {% if post.categories %} <small>categories: <em>{{ post.categories | join: "</em> - <em>" }}</em></small> {% endif %} {::comment} {% if post.tags %} <small>, tags: <em>{{ post.tags | join: "</em> - <em>" }}</em></small> {% endif %}{:/comment}

{% endfor %}
