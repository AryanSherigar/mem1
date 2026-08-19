import cmd
import sys
import uuid
from datetime import datetime, timezone

class InteractiveChat(cmd.Cmd):
    intro = "Welcome to the Context Memory Interactive Chat. Type help or ? to list commands.\n"
    prompt = "(chat) "

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.context_id = f"ctx-{uuid.uuid4()}"
        self.session_id = f"session-{uuid.uuid4()}"
        print(f"Started new chat with Context ID: {self.context_id}, Session ID: {self.session_id}")

    def do_chat(self, arg):
        """chat [message]: Send a message to the assistant."""
        if not arg:
            print("Please provide a message. Usage: chat [message]")
            return
            
        print("Assistant is typing...")
        try:
            reply = self.engine.generate_reply(self.context_id, self.session_id, arg)
            print(f"\nAssistant: {reply}\n")
        except NotImplementedError:
            print("\nAssistant: Engine wiring required. Mock response.\n")
        except Exception as e:
            print(f"\nError: {e}\n")

    def do_search(self, arg):
        """search [query]: Search extracted facts and context."""
        if not arg:
            print("Please provide a query. Usage: search [query]")
            return
            
        print(f"Searching memory for: '{arg}'...")
        try:
            now = datetime.now(timezone.utc)
            result = self.engine.search_memories(self.context_id, arg, now)
            print(f"\nSearch Result: {result}\n")
        except NotImplementedError:
            print("\nSearch Result: Engine wiring required.\n")
        except Exception as e:
            print(f"\nError: {e}\n")

    def do_context(self, arg):
        """context [context_id]: Switch to a different context ID."""
        if not arg:
            print(f"Current Context ID: {self.context_id}")
            return
        self.context_id = arg
        print(f"Switched context to: {self.context_id}")

    def do_exit(self, arg):
        """Exit the chat interface."""
        print("Goodbye!")
        return True

def main():
    try:
        from context_memory.core.logging import setup_logging
        import logging
        setup_logging(level=logging.INFO)
        from api.routes import get_engine
        print("Initializing Memory Engine...")
        engine = get_engine()
        InteractiveChat(engine).cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to initialize Memory Engine: {e}")
        print("Tip: Ensure PostgreSQL and HydraDB are running.")
        sys.exit(1)

if __name__ == "__main__":
    main()
