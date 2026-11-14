import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chatbot import create_chatbot

HELP = "Comandos: /perfil si | /perfil no | /borrar-perfil | /olvidar | /salir"


def main() -> None:
    bot = create_chatbot()
    user_id = input("Identificador local (Enter para usar modo anónimo): ").strip() or None
    session = bot.new_session(user_id=user_id)
    print(HELP)

    while True:
        text = input("\n> ").strip()
        if not text:
            continue
        if text == "/salir":
            break
        if text == "/olvidar":
            session.memory.clear()
            print("Memoria de la sesión borrada.")
            continue
        if text.startswith("/perfil"):
            if not user_id:
                print("El modo anónimo no tiene perfil.")
            else:
                bot.profiles.set_consent(user_id, text.endswith("si"))
                print("Perfil actualizado.")
            continue
        if text == "/borrar-perfil" and user_id:
            bot.profiles.forget(user_id)
            print("Perfil eliminado.")
            continue

        resp = bot.answer(session, text)
        print(f"\n{resp.answer}")
        for s in resp.sources:
            print(f"  [{s['id']}] {s['source']} (p. {s['page'] or '-'}, similitud {s['score']})")


if __name__ == "__main__":
    main()