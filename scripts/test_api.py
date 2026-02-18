import json
import urllib.request

url = "http://localhost:5000/api/sandbox/map_gfd_with_bayesian"
payload = json.dumps({"galaxy_id": "milky_way"}).encode()
req = urllib.request.Request(url, data=payload,
                              headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
d = json.loads(resp.read())

c = d.get("chart", {})
print("gfd_base:", len(c.get("gfd_base", [])), "points")
print("gfd_photometric:", len(c.get("gfd_photometric", [])), "points")
print("gfd_covariant:", len(c.get("gfd_covariant", [])), "points")

sf = d.get("sigma_fit", {})
print()
print("SIGMA FIT:")
print("  sigma =", sf.get("sigma"))
print("  sigma^2 =", sf.get("sigma_squared"))
print("  RMS base (no vortex) =", sf.get("rms_base"), "km/s")
print("  RMS covariant (with vortex) =", sf.get("rms"), "km/s")
print("  Improvement =", sf.get("improvement"), "%")
print()
print("Bayesian RMS =", d.get("rms"), "km/s")
