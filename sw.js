// Service worker : met en cache la base de mots et l'interface pour que
// l'outil demarre instantanement et fonctionne hors connexion apres la
// premiere visite. Les fichiers de donnees sont demandes avec ?v=N
// (DATA_VERSION dans index.html) : regenerer la base et incrementer N
// suffit a invalider le cache.

const CACHE_NAME = "chercheMots-v1";
const DATA_FILES = ["words_meta.json", "vectors.bin"];

self.addEventListener("install", () => {
	self.skipWaiting();
});

self.addEventListener("activate", event => {
	event.waitUntil(
		caches.keys()
			.then(keys => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k))))
			.then(() => self.clients.claim())
	);
});

self.addEventListener("fetch", event => {
	const url = new URL(event.request.url);
	const isDataFile = DATA_FILES.some(name => url.pathname.endsWith("/" + name));
	const isPage = event.request.mode === "navigate" || url.pathname.endsWith("/index.html");

	if (isDataFile) {
		// Donnees : cache d'abord (lourdes et versionnees par ?v=N)
		event.respondWith(
			caches.open(CACHE_NAME).then(async cache => {
				const cached = await cache.match(event.request);
				if (cached) return cached;
				const response = await fetch(event.request);
				if (response.ok) cache.put(event.request, response.clone());
				return response;
			})
		);
	} else if (isPage) {
		// Interface : reseau d'abord (pour recevoir les mises a jour),
		// repli sur le cache hors connexion
		event.respondWith(
			fetch(event.request)
				.then(response => {
					if (response.ok) {
						const copy = response.clone();
						caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
					}
					return response;
				})
				.catch(() => caches.match(event.request))
		);
	}
});
