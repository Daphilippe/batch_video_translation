import json
import time
import random
import logging
from pathlib import Path
from deep_translator import GoogleTranslator
from modules.translator import BaseTranslator
from utils.srt_handler import SRTHandler

logger = logging.getLogger(__name__)

class LegacyTranslator(BaseTranslator):
    def __init__(self, input_dir, output_dir, config_path="configs/settings.json"):
        # Chargement de la config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        super().__init__(input_dir, output_dir, None, extensions=(".srt",))
        
        # Initialisation du traducteur
        self.translator = GoogleTranslator(
            source=self.config["translation"]["source_lang"], 
            target=self.config["translation"]["target_lang"]
        )
        
        # Préparation du dictionnaire (trié par longueur décroissante)
        raw_dict = self.config["technical_dictionary"]
        self.tech_dict = sorted(raw_dict.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Gestion du cache
        self.cache_path = Path(self.config["translation"]["cache_file"])
        self.cache = self._load_cache()
        self.name="Legacy translation"

    def _load_cache(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_cache(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _apply_dictionary(self, text: str) -> str:
        """Applique les termes techniques avant traduction."""
        t = text.lower()
        for key, val in self.tech_dict:
            t = t.replace(key, f"({val})")
        return t

    def _safe_translate_batch(self, batch: list) -> list:
        """Traduction sécurisée avec gestion du rate-limit (429)."""
        query = " ||| ".join(batch)
        try:
            result = self.translator.translate(query)
            if not result: return []
            return [res.strip() for res in result.split(" ||| ")]
        except Exception as e:
            if "429" in str(e):
                delay = self.config["translation"]["retry_delay"]
                logger.warning(f"Rate limit hit. Waiting {delay}s...")
                time.sleep(delay)
                return self._safe_translate_batch(batch)
            logger.error(f"Translation error: {e}")
            return []

    def translate_logic(self, text: str):
        """Découpe le SRT, vérifie le cache et traduit par batchs."""
        lines = text.splitlines()
        final_lines = []
        to_translate = [] # Liste de (index_dans_final_lines, texte_a_traduire, hash)

        # 1. Analyse du fichier et vérification du cache
        for line in lines:
            clean = line.strip()
            if clean.isdigit() or "-->" in clean or not clean:
                final_lines.append(line)
            else:
                l_hash = SRTHandler.get_hash(clean)
                if l_hash in self.cache:
                    final_lines.append(self.cache[l_hash])
                else:
                    pre_treated = self._apply_dictionary(clean)
                    to_translate.append((len(final_lines), pre_treated, l_hash))
                    final_lines.append(None) # Placeholder

        # 2. Traduction par batchs pour respecter les limites
        i = 0
        max_chars = self.config["translation"]["max_chars_batch"]
        while i < len(to_translate):
            current_batch, current_meta, current_len = [], [], 0
            
            while i < len(to_translate) and current_len < max_chars:
                idx, txt, h = to_translate[i]
                current_batch.append(txt)
                current_meta.append((idx, h))
                current_len += len(txt)
                i += 1
            
            if current_batch:
                results = self._safe_translate_batch(current_batch)
                for (idx, h), res in zip(current_meta, results):
                    final_lines[idx] = res
                    self.cache[h] = res
                
                time.sleep(random.uniform(1.2, 2.5)) # Simulation humaine

        return "\n".join([l if l is not None else "..." for l in final_lines])