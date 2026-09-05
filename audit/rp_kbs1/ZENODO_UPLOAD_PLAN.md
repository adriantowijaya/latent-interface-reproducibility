# Zenodo v1.1.0 Upload Plan

1. Open the published v1.0.0 Zenodo record.
2. Choose **New version**.
3. Import/copy the files from the previous version.
4. Preserve the existing v1.0.0 checkpoint archives and foundation manifests.
5. Add the KBS extension archive `kbs-posthoc-extension-v1.1.0.zip`.
6. Preferably add a fresh full GitHub source snapshot from tag `v1.1.0` named `latent-interface-reproducibility-source-v1.1.0.zip`.
7. If the old source-v1.0.0 ZIP remains in the imported file set, keep it as historical foundation only; the v1.1.0 source snapshot should be the current code/navigation authority.
8. Update Zenodo metadata using `ZENODO_METADATA_v1.1.0_DRAFT.md`.
9. Preview the record and verify creators, version, title, files, licenses, GitHub relation, and visibility.
10. Publish the new version.
11. Record the new **version-specific DOI** exactly as assigned by Zenodo.
12. Rebind the KBS manuscript using the supplied manuscript rebinding template/script.
13. Verify the DOI resolves and that the GitHub release/tag is public before submission.

Do not invent `.v2` or `.v1.1.0` suffixes on the old DOI. Zenodo assigns a distinct DOI to every published version and links it to the same concept record.
